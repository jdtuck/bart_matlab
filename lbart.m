function out = lbart(x_train, y_train, x_test, opts)
%LBART BART for a binary outcome with a logistic link.
%
%   out = lbart(x_train, y_train)
%   out = lbart(x_train, y_train, x_test, opts)
%
% Model
%   P(Y = 1 | x) = 1 / (1 + exp(-(mu0 + f(x)))),  mu0 = logit(mean(y))
% fitted with Polson, Scott & Windle (2013) Polya-Gamma augmentation: draw
% omega_i ~ PG(1, mu0 + f(x_i)), then the working response (y_i - 1/2)/omega_i
% is Gaussian around mu0 + f(x_i) with variance 1/omega_i, which is exactly the
% weighted sum-of-trees problem BARTCORE solves.  The leaf prior standard
% deviation is 4/(k*sqrt(ntree)), reflecting the wider logistic scale.
%
% Note: the Polya-Gamma draws use a truncated series (see BARTPOLYAGAMMARND)
% and are therefore approximate.  PBART uses an exact augmentation scheme; use
% LBART when odds-ratio interpretability of the latent scale matters more.
%
% Output mirrors PBART: prob_train, prob_test and their posterior means, plus
% draws on the logit scale in yhat_train / yhat_test.
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), JSS 97(1).
% doi:10.18637/jss.v097.i01

if nargin < 3, x_test = []; end
if nargin < 4, opts   = struct(); end
opts    = bartOptions(opts);
pgterms = opts.pgterms;

y_train = double(y_train(:));
n       = size(x_train, 1);
lev     = unique(y_train);
if numel(lev) == 2 && any(~ismember(y_train, [0 1]))
    y_train = double(y_train == lev(2));
elseif numel(lev) > 2
    error('lbart:binary', 'y_train must be binary.');
end

cuts = bartMakeCuts(x_train, opts.numcut, opts.usequants);
Xb   = bartBinX(x_train, cuts);
ncut = cellfun(@numel, cuts);
if isempty(x_test)
    Xbtest = [];
else
    Xbtest = bartBinX(x_test, cuts);
end

pbar   = min(max(mean(y_train), 1 / (n + 1)), 1 - 1 / (n + 1));
offset = log(pbar / (1 - pbar));
tau    = 4 / (opts.k * sqrt(opts.ntree));
kappa  = y_train - 0.5;

copts             = opts;
copts.sigma       = 1;
copts.sampleSigma = false;
copts.lambda      = 1;
copts.Xbtest      = Xbtest;
copts.latentfun   = @(f) logitLatent(f, kappa, offset, pgterms);

core = bartCore(Xb, ncut, zeros(n, 1), ones(n, 1), tau, copts);

out = struct();
out.type     = 'lbart';
out.offset   = offset;
out.tau      = tau;
out.cuts     = cuts;
out.trees    = core.trees;
out.varcount = core.varcount;
out.varprob  = mean(core.varcount ./ max(sum(core.varcount, 2), 1), 1);
out.accept   = core.accept;
out.opts     = opts;

yhat_train = offset + core.fhat_train;
prob_train = 1 ./ (1 + exp(-yhat_train));
out.prob_train_mean = mean(prob_train, 1);
if isempty(x_test)
    yhat_test = []; prob_test = [];
    out.prob_test_mean = [];
else
    yhat_test = offset + core.fhat_test;
    prob_test = 1 ./ (1 + exp(-yhat_test));
    out.prob_test_mean = mean(prob_test, 1);
end
if opts.savedraws
    out.yhat_train = yhat_train; out.prob_train = prob_train;
    out.yhat_test  = yhat_test;  out.prob_test  = prob_test;
else
    out.yhat_train = []; out.prob_train = [];
    out.yhat_test  = []; out.prob_test  = [];
end
end

function [ytree, w] = logitLatent(f, kappa, offset, pgterms)
eta   = offset + f;
w     = bartPolyaGammaRnd(eta, pgterms);
ytree = kappa ./ w - offset;
end
