function out = wbart(x_train, y_train, x_test, opts)
%WBART BART for a continuous outcome (Gaussian errors).
%
%   out = wbart(x_train, y_train)
%   out = wbart(x_train, y_train, x_test)
%   out = wbart(x_train, y_train, x_test, opts)
%
% Model
%   y_i = mu0 + f(x_i) + eps_i,   eps_i ~ N(0, sigma^2 / w_i)
% where mu0 = mean(y) is a fixed centering offset, f is a sum of ntree trees,
% and sigma^2 has the scaled inverse chi-square prior calibrated so that
% P(sigma < sigest) = q, with sigest the residual standard deviation from a
% least-squares fit (or sd(y) when p >= n).  The leaf prior standard deviation
% is range(y) / (2*k*sqrt(ntree)), so k controls shrinkage of the fit.
%
% Inputs
%   x_train  n-by-p numeric covariate matrix (dummy-code factors first, e.g.
%            with BARTMODELMATRIX)
%   y_train  n-by-1 continuous outcome
%   x_test   optional matrix of covariates to predict at ([] for none)
%   opts     see BARTOPTIONS
%
% Output
%   out.yhat_train        ndpost-by-n posterior draws of mu0 + f(x_i)
%   out.yhat_train_mean   posterior mean, 1-by-n
%   out.yhat_test         draws at x_test (empty if x_test is empty)
%   out.yhat_test_mean    posterior mean at x_test
%   out.sigma             ndpost-by-1 draws of the error sd
%   out.varcount          ndpost-by-p covariate split counts
%   out.varprob           posterior mean proportion of splits per covariate
%   out.offset, .cuts, .trees, .type  everything BARTPREDICT needs
%
% Example
%   n = 500; x = rand(n,5);
%   y = 10*sin(pi*x(:,1).*x(:,2)) + 20*(x(:,3)-0.5).^2 + 10*x(:,4) + 5*x(:,5);
%   f = wbart(x, y + randn(n,1), [], bartOptions('ntree',50,'ndpost',500));
%   corr(f.yhat_train_mean(:), y)
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), JSS 97(1).
% doi:10.18637/jss.v097.i01

if nargin < 3, x_test = []; end
if nargin < 4, opts   = struct(); end
opts = bartOptions(opts);

y_train = y_train(:);
[n, p]  = size(x_train);
if numel(y_train) ~= n
    error('wbart:dims', 'y_train must have %d elements.', n);
end

w = opts.w;
if isempty(w), w = ones(n, 1); else w = w(:); end

% ---- cutpoints and binning ----------------------------------------------
cuts = bartMakeCuts(x_train, opts.numcut, opts.usequants);
Xb   = bartBinX(x_train, cuts);
ncut = cellfun(@numel, cuts);
if isempty(x_test)
    Xbtest = [];
else
    Xbtest = bartBinX(x_test, cuts);
end

% ---- priors --------------------------------------------------------------
offset = mean(y_train);
ytr    = y_train - offset;
tau    = (max(y_train) - min(y_train)) / (2 * opts.k * sqrt(opts.ntree));
if tau <= 0
    error('wbart:constant', 'y_train is constant.');
end

sigest = opts.sigest;
if isempty(sigest)
    if n > p + 1
        Xd  = [ones(n, 1) x_train];
        bhat = Xd \ y_train;
        res  = y_train - Xd * bhat;
        sigest = sqrt(sum(w .* res.^2) / (n - p - 1));
    else
        sigest = std(y_train);
    end
    if ~isfinite(sigest) || sigest <= 0
        sigest = std(y_train);
    end
end
% Solve P(sigma < sigest) = q for the prior scale lambda.
qchi   = 2 * gammaincinv(1 - opts.q, opts.nu / 2);
lambda = sigest^2 * qchi / opts.nu;

copts             = opts;
copts.sigma       = sigest;
copts.sampleSigma = true;
copts.lambda      = lambda;
copts.latentfun   = [];
copts.Xbtest      = Xbtest;

% ---- run -----------------------------------------------------------------
core = bartCore(Xb, ncut, ytr, w, tau, copts);

% ---- package -------------------------------------------------------------
out = struct();
out.type      = 'wbart';
out.offset    = offset;
out.tau       = tau;
out.sigest    = sigest;
out.lambda    = lambda;
out.cuts      = cuts;
out.trees     = core.trees;
out.sigma     = core.sigma;
out.varcount  = core.varcount;
out.varprob   = mean(core.varcount ./ max(sum(core.varcount, 2), 1), 1);
out.accept    = core.accept;
out.opts      = opts;

out.yhat_train      = offset + core.fhat_train;
out.yhat_train_mean = mean(out.yhat_train, 1);
if isempty(x_test)
    out.yhat_test = []; out.yhat_test_mean = [];
else
    out.yhat_test      = offset + core.fhat_test;
    out.yhat_test_mean = mean(out.yhat_test, 1);
end
if ~opts.savedraws
    out.yhat_train = [];
    out.yhat_test  = [];
end
end
