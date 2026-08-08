function out = pbart(x_train, y_train, x_test, opts)
%PBART BART for a binary outcome with a probit link.
%
%   out = pbart(x_train, y_train)
%   out = pbart(x_train, y_train, x_test, opts)
%
% Model
%   P(Y = 1 | x) = Phi(mu0 + f(x)),   mu0 = Phi^{-1}(mean(y))
% fitted with Albert & Chib (1993) data augmentation: latent
% z_i ~ N(mu0 + f(x_i), 1) truncated to be positive when y_i = 1 and negative
% when y_i = 0.  Given z the model is a Gaussian sum-of-trees with sigma fixed
% at 1, so no variance parameter is sampled.  The leaf prior standard deviation
% is 3/(k*sqrt(ntree)), which keeps mu0 + f on the scale where Phi is
% informative.
%
% Inputs
%   x_train  n-by-p numeric covariates
%   y_train  n-by-1 outcome coded 0/1 (logical or two-level numeric is fine)
%   x_test   optional prediction covariates
%   opts     see BARTOPTIONS.  ntree = 50 is a good default for binary BART.
%
% Output
%   out.prob_train / out.prob_test        ndpost-by-n draws of P(Y=1|x)
%   out.prob_train_mean / _test_mean      posterior means
%   out.yhat_train / out.yhat_test        draws on the probit (linear) scale
%   out.varcount, out.varprob, out.offset, out.cuts, out.trees, out.type
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), JSS 97(1).
% doi:10.18637/jss.v097.i01

if nargin < 3, x_test = []; end
if nargin < 4, opts   = struct(); end
opts = bartOptions(opts);

y_train = double(y_train(:));
n       = size(x_train, 1);
if numel(y_train) ~= n
    error('pbart:dims', 'y_train must have %d elements.', n);
end
lev = unique(y_train);
if numel(lev) > 2 || any(~ismember(y_train, [0 1]))
    if numel(lev) == 2
        y_train = double(y_train == lev(2));
    else
        error('pbart:binary', 'y_train must be binary.');
    end
end

cuts = bartMakeCuts(x_train, opts.numcut, opts.usequants);
Xb   = bartBinX(x_train, cuts);
ncut = cellfun(@numel, cuts);
if isempty(x_test)
    Xbtest = [];
else
    Xbtest = bartBinX(x_test, cuts);
end

pbar   = mean(y_train);
pbar   = min(max(pbar, 1 / (n + 1)), 1 - 1 / (n + 1));   % guard 0 and 1
offset = bartPhiInv(pbar);
tau    = 3 / (opts.k * sqrt(opts.ntree));

isone = (y_train == 1);
copts             = opts;
copts.sigma       = 1;
copts.sampleSigma = false;
copts.lambda      = 1;
copts.Xbtest      = Xbtest;
copts.latentfun   = @(f) probitLatent(f, isone, offset);

core = bartCore(Xb, ncut, zeros(n, 1), ones(n, 1), tau, copts);

out = struct();
out.type     = 'pbart';
out.offset   = offset;
out.tau      = tau;
out.cuts     = cuts;
out.trees    = core.trees;
out.varcount = core.varcount;
out.varprob  = mean(core.varcount ./ max(sum(core.varcount, 2), 1), 1);
out.accept   = core.accept;
out.opts     = opts;

yhat_train        = offset + core.fhat_train;
prob_train        = bartPhi(yhat_train);
out.prob_train_mean = mean(prob_train, 1);
if isempty(x_test)
    yhat_test = []; prob_test = [];
    out.prob_test_mean = [];
else
    yhat_test = offset + core.fhat_test;
    prob_test = bartPhi(yhat_test);
    out.prob_test_mean = mean(prob_test, 1);
end
if opts.savedraws
    out.yhat_train = yhat_train;
    out.prob_train = prob_train;
    out.yhat_test  = yhat_test;
    out.prob_test  = prob_test;
else
    out.yhat_train = []; out.prob_train = [];
    out.yhat_test  = []; out.prob_test  = [];
end
end

function [ytree, w] = probitLatent(f, isone, offset)
z     = bartTruncNormRnd(offset + f, isone);
ytree = z - offset;
w     = ones(numel(z), 1);
end
