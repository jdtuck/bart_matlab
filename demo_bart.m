%% demo_bart -- BART for continuous, binary, categorical and survival outcomes
%
% Run this file from the folder containing the BART .m files.  Each section
% simulates data with a known truth, fits the corresponding model, and reports
% how well the posterior recovers that truth.  Set DOPLOT = true for figures.
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), "Nonparametric Machine
% Learning and Efficient Computation with Bayesian Additive Regression Trees:
% The BART R Package", Journal of Statistical Software 97(1).
% doi:10.18637/jss.v097.i01

close all; more off;
DOPLOT = true;
rand('state', 2024); randn('state', 2024);          %#ok<RAND>

% Small settings so the demo runs in a couple of minutes.  For real work use
% the defaults (ntree = 200 for wbart, 50 for the binary/survival models,
% ndpost = 1000, nskip = 100 or more).
opts = bartOptions('ntree', 50, 'ndpost', 500, 'nskip', 250, 'printevery', 0);

%% ---------------------------------------------------------------- CONTINUOUS
% Friedman's test function: nonlinear, interacting, with 5 irrelevant columns.
fprintf('\n===== 1. Continuous outcome (wbart) =====\n');
n = 500; p = 10; sigma_true = 1.0;
f0 = @(x) 10*sin(pi*x(:,1).*x(:,2)) + 20*(x(:,3)-0.5).^2 + 10*x(:,4) + 5*x(:,5);

x  = rand(n, p);
y  = f0(x) + sigma_true * randn(n, 1);
xt = rand(500, p);

tic; fit1 = wbart(x, y, xt, opts); t = toc;

fprintf('fitted in %.1f s\n', t);
fprintf('sigma: posterior mean %.3f, 95%% CI [%.3f %.3f], truth %.2f\n', ...
        mean(fit1.model.sigma), bartPctl(fit1.model.sigma, 2.5), bartPctl(fit1.model.sigma, 97.5), sigma_true);
fprintf('out-of-sample RMSE for f(x): %.3f   (sd of f is %.2f)\n', ...
        sqrt(mean((fit1.model.yhat_test_mean(:) - f0(xt)).^2)), std(f0(xt)));
fprintf('share of splits per covariate (x1..x10):\n   %s\n', ...
        sprintf('%.3f ', fit1.model.varprob));
fprintf('-> the five relevant covariates take most of the splits\n');

% Posterior credible intervals cover the truth at roughly the nominal rate.
lo = bartPctl(fit1.model.yhat_test, 2.5);
hi = bartPctl(fit1.model.yhat_test, 97.5);
fprintf('coverage of pointwise 95%% intervals for f(x): %.3f\n', ...
        mean(f0(xt) >= lo(:) & f0(xt) <= hi(:)));

if DOPLOT
    figure; plot(f0(xt), fit1.model.yhat_test_mean, '.'); hold on;
    plot(xlim, xlim, 'r-'); xlabel('true f(x)'); ylabel('posterior mean');
    title('wbart: out-of-sample fit');
end

%% -------------------------------------------------------------------- BINARY
fprintf('\n===== 2. Binary outcome (pbart, probit) =====\n');
n = 800;
x = rand(n, 5);
eta_true = 3*(x(:,1) - 0.5) + 2*sin(3*x(:,2)) - 1;    % x3..x5 irrelevant
p_true   = bartPhi(eta_true);
yb       = double(rand(n, 1) < p_true);

tic; fit2 = pbart(x, yb, x, opts); t = toc;

fprintf('fitted in %.1f s (%d events of %d)\n', t, sum(yb), n);
fprintf('RMSE for P(Y=1|x): %.3f   correlation %.3f\n', ...
        sqrt(mean((fit2.prob_train_mean(:) - p_true).^2)), ...
        corrPearson(fit2.prob_train_mean(:), p_true));
fprintf('misclassification rate at the 0.5 cutoff: %.3f (Bayes rate %.3f)\n', ...
        mean((fit2.prob_train_mean(:) > 0.5) ~= yb), ...
        mean((p_true > 0.5) ~= yb));
fprintf('share of splits per covariate:\n   %s\n', sprintf('%.3f ', fit2.varprob));

% The same data with a logit link (Polya-Gamma augmentation).
fit2b = lbart(x, yb, [], opts);
fprintf('lbart RMSE for P(Y=1|x): %.3f\n', ...
        sqrt(mean((fit2b.prob_train_mean(:) - p_true).^2)));

%% --------------------------------------------------------------- CATEGORICAL
fprintf('\n===== 3. Categorical outcome (mbart, 3 levels) =====\n');
n = 800;
x   = rand(n, 4);
lin = [3*x(:,1), 3*x(:,2).^2, ones(n,1)];             % level 3 is the baseline
P   = exp(lin); P = P ./ sum(P, 2);
u   = rand(n, 1);
yc  = 1 + (u > P(:,1)) + (u > P(:,1) + P(:,2));

tic; fit3 = mbart(x, yc, x, opts); t = toc;

fprintf('fitted in %.1f s; level counts: %s\n', t, ...
        sprintf('%d ', accumarray(yc, 1)));
fprintf('mean abs error of the category probabilities: %.3f\n', ...
        mean(mean(abs(fit3.prob_train_mean - P))));
fprintf('probabilities sum to one: max deviation %.2e\n', ...
        max(abs(sum(fit3.prob_train_mean, 2) - 1)));
[~, bayes] = max(P, [], 2);
fprintf('agreement with the Bayes classifier: %.3f\n', ...
        mean(fit3.pred_train(:) == bayes));

%% ------------------------------------------------------------- TIME-TO-EVENT
fprintf('\n===== 4. Time-to-event outcome (survbart) =====\n');
n = 600;
x    = rand(n, 3);
rate = 0.5 * exp(1.5 * x(:,1));                       % x2, x3 irrelevant
Tev  = -log(rand(n,1)) ./ rate;                       % exponential times
Cens = 0.5 + 3 * rand(n, 1);                          % random censoring
times = min(Tev, Cens);
delta = double(Tev <= Cens);

xtest = [0.1 0.5 0.5; 0.5 0.5 0.5; 0.9 0.5 0.5];      % low / mid / high risk
sopts = bartOptions(opts, 'K', 25);                   % coarsen the time grid

tic; fit4 = survbart(times, delta, x, xtest, sopts); t = toc;

fprintf('fitted in %.1f s (%d events, %d time points, %d person-period rows)\n', ...
        t, sum(delta), fit4.K, fit4.nlong);
S_true = @(x1, tt) exp(-0.5 * exp(1.5 * x1) * tt);
fprintf('\n   x1    t     S true   S BART   95%% CI\n');
show = round(linspace(2, fit4.K, 4));
for i = 1:size(xtest, 1)
    for k = show
        fprintf('  %.1f  %5.2f   %.3f    %.3f    [%.3f %.3f]\n', ...
                xtest(i,1), fit4.times(k), S_true(xtest(i,1), fit4.times(k)), ...
                fit4.surv_mean(i,k), fit4.surv_lower(i,k), fit4.surv_upper(i,k));
    end
end
mae = zeros(size(xtest,1), 1);
for i = 1:size(xtest,1)
    mae(i) = mean(abs(fit4.surv_mean(i,:) - S_true(xtest(i,1), fit4.times)'));
end
fprintf('\nmean absolute error of the survival curves: %s\n', sprintf('%.3f ', mae));
fprintf('-> no proportional-hazards assumption was used; the trees learn both\n');
fprintf('   the baseline hazard and the covariate effect from the time grid.\n');

if DOPLOT
    figure; hold on;
    cols = {'b', 'k', 'r'};
    for i = 1:size(xtest,1)
        stairs(fit4.times, fit4.surv_mean(i,:), cols{i});
        stairs(fit4.times, S_true(xtest(i,1), fit4.times), [cols{i} '--']);
    end
    xlabel('time'); ylabel('S(t|x)');
    title('survbart: posterior mean (solid) vs truth (dashed)');
end

%% ----------------------------------------------------- prediction at new data
fprintf('\n===== 5. Predicting from stored posterior trees =====\n');
xnew = rand(4, 10);
pr   = bartPredict(fit1.model, xnew);
fprintf('posterior mean of f at 4 new points: %s\n', sprintf('%.2f ', pr.yhat_mean));
fprintf('posterior sd at those points:        %s\n', sprintf('%.2f ', std(pr.yhat, 0, 1)));
fprintf('\ndone.\n');

% (helper functions live in bartPctl.m and corrPearson.m so that this file
%  runs as a plain script in both MATLAB and Octave)
