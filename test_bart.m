% test_bart.m -- correctness checks
more off;
rand('state', 7); randn('state', 7);

fprintf('=== 1. utilities ===\n');
g = bartGamrnd(repmat(3, 20000, 1));
fprintf('Gamma(3,1): mean %.3f (3)  var %.3f (3)\n', mean(g), var(g));
g = bartGamrnd(repmat(0.4, 20000, 1));
fprintf('Gamma(0.4,1): mean %.3f (0.4)  var %.3f (0.4)\n', mean(g), var(g));

z = bartTruncNormRnd(repmat(0.5, 20000, 1), true(20000,1));
fprintf('TN(0.5,1)+ : mean %.3f  min %.4f (>0)\n', mean(z), min(z));
z = bartTruncNormRnd(repmat(-3, 20000, 1), true(20000,1));
fprintf('TN(-3,1)+ (tail): mean %.3f (0.283)  min %.4f\n', mean(z), min(z));
z = bartTruncNormRnd(repmat(1, 20000, 1), false(20000,1));
fprintf('TN(1,1)-  : mean %.3f (-0.525)  max %.4f (<0)\n', mean(z), max(z));

om = bartPolyaGammaRnd(repmat(2, 20000, 1));
% E[PG(1,c)] = tanh(c/2)/(2c)
fprintf('PG(1,2): mean %.4f  target %.4f\n', mean(om), tanh(1)/4);
om = bartPolyaGammaRnd(zeros(20000,1));
fprintf('PG(1,0): mean %.4f  target 0.25\n', mean(om));

fprintf('Phi/PhiInv roundtrip max err: %.2e\n', ...
        max(abs(bartPhiInv(bartPhi(linspace(-4,4,101))) - linspace(-4,4,101))));

fprintf('\n=== 2. cutpoints / binning / assignment ===\n');
X = [ (1:10)', [0;0;1;1;0;1;1;0;1;0] ];
cuts = bartMakeCuts(X, 100, false);
fprintf('ncut = [%d %d] (expect 9 1)\n', numel(cuts{1}), numel(cuts{2}));
Xb = bartBinX(X, cuts);
fprintf('code col1 = %s\n', mat2str(Xb(:,1)'));
fprintf('code col2 = %s\n', mat2str(Xb(:,2)'));
% hand-built stump: split on var 1 at cut 5 -> x<=cuts{1}(5)=5.5
t = struct('var',[1;0;0],'cut',[5;0;0],'lc',[2;0;0],'rc',[3;0;0],'mu',[0;-1;1]);
nid = bartAssign(t, Xb);
fprintf('assign: %s (expect 2 for x<=5)\n', mat2str(nid'));

fprintf('\n=== 3. wbart on Friedman data ===\n');
n = 400; p = 10;
f0 = @(x) 10*sin(pi*x(:,1).*x(:,2)) + 20*(x(:,3)-0.5).^2 + 10*x(:,4) + 5*x(:,5);
x  = rand(n,p);  ytrue = f0(x);  y = ytrue + 1.0*randn(n,1);
xt = rand(200,p); yt = f0(xt);
o  = bartOptions('ntree',50,'ndpost',400,'nskip',200,'printevery',0);
tic; m1 = wbart(x, y, xt, o); t1 = toc;
fprintf('time %.1fs  accept G/P/C = %.2f %.2f %.2f\n', t1, m1.model.accept);
fprintf('train corr %.3f  RMSE(f) %.3f  (sd(f0)=%.2f)\n', ...
        corr(m1.model.yhat_train_mean(:), ytrue), ...
        sqrt(mean((m1.model.yhat_train_mean(:)-ytrue).^2)), std(ytrue));
fprintf('test  corr %.3f  RMSE(f) %.3f\n', ...
        corr(m1.model.yhat_test_mean(:), yt), sqrt(mean((m1.model.yhat_test_mean(:)-yt).^2)));
fprintf('sigma post mean %.3f (true 1.0)\n', mean(m1.model.sigma));
fprintf('varprob first 6: %s\n', mat2str(round(1000*m1.model.varprob(1:6))/1000));

% bartPredict must reproduce the in-sample test fit
pr = bartPredict(m1.model, xt);
fprintf('bartPredict vs stored test fit, max abs diff: %.2e\n', ...
        max(abs(pr.yhat_mean - m1.model.yhat_test_mean)));

fprintf('\n=== 4. linear check + sparse prior ===\n');
n = 300; x = randn(n,20); y = 2*x(:,1) - 1.5*x(:,2) + 0.5*randn(n,1);
o = bartOptions('ntree',50,'ndpost',300,'nskip',200,'printevery',0,'sparse',true);
m2 = wbart(x, y, [], o);
fprintf('corr with truth %.3f, sigma %.3f (true 0.5)\n', ...
        corr(m2.model.yhat_train_mean(:), 2*x(:,1)-1.5*x(:,2)), mean(m2.model.sigma));
[~, top] = sort(m2.model.varprob, 'descend');
fprintf('top 3 variables (sparse): %s (expect 1 and 2)\n', mat2str(top(1:3)));

fprintf('\n=== 5. pbart / lbart ===\n');
n = 600; x = rand(n,5);
eta = 3*(x(:,1)-0.5) + 2*sin(3*x(:,2)) - 1;
pt = bartPhi(eta); yb = double(rand(n,1) < pt);
o = bartOptions('ntree',50,'ndpost',400,'nskip',200,'printevery',0);
m3 = pbart(x, yb, x, o);
fprintf('pbart: corr(phat,ptrue) %.3f  RMSE %.3f  err rate %.3f\n', ...
        corr(m3.prob_train_mean(:), pt), ...
        sqrt(mean((m3.prob_train_mean(:)-pt).^2)), ...
        mean((m3.prob_train_mean(:)>0.5) ~= yb));
ptl = 1./(1+exp(-eta)); ybl = double(rand(n,1) < ptl);
m4 = lbart(x, ybl, [], o);
fprintf('lbart: corr(phat,ptrue) %.3f  RMSE %.3f\n', ...
        corr(m4.prob_train_mean(:), ptl), ...
        sqrt(mean((m4.prob_train_mean(:)-ptl).^2)));

fprintf('\n=== 6. mbart (3 categories) ===\n');
n = 600; x = rand(n,3);
lin = [2*x(:,1), 2*x(:,2), ones(n,1)*0.8];
P   = exp(lin); P = P ./ sum(P,2);
u = rand(n,1); yc = 1 + (u > P(:,1)) + (u > P(:,1)+P(:,2));
o = bartOptions('ntree',50,'ndpost',300,'nskip',200,'printevery',0);
m5 = mbart(x, yc, x(1:5,:), o);
fprintf('levels: %s\n', mat2str(m5.levels'));
fprintf('row sums of prob (should be 1): %.6f %.6f\n', ...
        min(sum(m5.prob_train_mean,2)), max(sum(m5.prob_train_mean,2)));
fprintf('mean abs err vs true probs: %.3f\n', mean(mean(abs(m5.prob_train_mean - P))));
fprintf('classification agreement with argmax(P): %.3f\n', ...
        mean(m5.pred_train(:) == argmaxrow(P)));
prm = bartPredict(m5, x(1:5,:));
fprintf('bartPredict(mbart) vs stored, max diff %.2e\n', ...
        max(max(abs(prm.prob_mean - m5.prob_test_mean))));

fprintf('\n=== 7. survbart ===\n');
n = 400; x = rand(n,3);
lam = 0.5*exp(1.5*x(:,1));                 % Weibull-ish: exponential rate
T = -log(rand(n,1))./lam;
C = 3*rand(n,1)+0.5;
tt = min(T,C); dd = double(T<=C);
fprintf('observed events: %d of %d\n', sum(dd), n);
xtest = [0.1 0.5 0.5; 0.9 0.5 0.5];
o = bartOptions('ntree',50,'ndpost',300,'nskip',200,'printevery',0,'K',20);
tic; m6 = survbart(tt, dd, x, xtest, o); t6 = toc;
fprintf('time %.1fs  K=%d  long rows=%d\n', t6, m6.K, m6.nlong);
strue = @(xx,t) exp(-0.5*exp(1.5*xx).*t);
for i = 1:2
    est = m6.surv_mean(i,:);
    tru = strue(xtest(i,1), m6.times)';
    fprintf('x1=%.1f: mean abs err %.3f;  S(1) est %.3f true %.3f\n', ...
            xtest(i,1), mean(abs(est-tru)), ...
            interp1(m6.times, est, min(1,max(m6.times)), 'linear'), ...
            strue(xtest(i,1), min(1,max(m6.times))));
end
fprintf('monotone survival: %d violations\n', sum(sum(diff(m6.surv_mean,1,2) > 1e-12)));
fprintf('bands ordered: %d violations\n', ...
        sum(sum(m6.surv_lower > m6.surv_upper)));

fprintf('\n=== 8. bartModelMatrix ===\n');
[Xm, inf1] = bartModelMatrix({ (1:6)', {'a','b','a','c','b','a'} });
disp(Xm); disp(inf1.names);
Xm2 = bartModelMatrix({ (7:8)', {'c','a'} }, inf1);
disp(Xm2);

fprintf('\nALL TESTS COMPLETED\n');
