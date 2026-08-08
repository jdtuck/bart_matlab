function out = survbart(times, delta, x_train, x_test, opts)
%SURVBART BART for a right-censored time-to-event outcome.
%
%   out = survbart(times, delta, x_train)
%   out = survbart(times, delta, x_train, x_test, opts)
%
% Implements the discrete-time (grouped hazard) representation of Sparapani et
% al.: the follow-up is split at the distinct event times t_1 < ... < t_K, and
% each subject contributes one Bernoulli observation per interval in which it
% is still at risk,
%
%   y_ik = 1 if subject i has the event in interval k, 0 otherwise,
%          k = 1, ..., k_i  where t_{k_i} is the interval containing its
%          event or censoring time,
%
%   P(y_ik = 1) = p(t_k, x_i) = Phi(mu0 + f(t_k, x_i)).
%
% Because the survival likelihood factorizes into these Bernoulli terms, the
% whole problem reduces to a single probit BART (PBART) on the long data set
% with time entered as an ordinary covariate (the first column), which lets the
% trees discover a fully nonparametric baseline hazard and arbitrary
% time-by-covariate interactions -- no proportional-hazards assumption.
% Survival is then read off as
%
%   S(t_k | x) = prod_{l <= k} (1 - p(t_l, x)).
%
% Inputs
%   times    n-by-1 observed follow-up time (event or censoring)
%   delta    n-by-1 event indicator, 1 = event, 0 = right censored
%   x_train  n-by-p covariates
%   x_test   optional covariates to predict at; defaults to x_train
%   opts     see BARTOPTIONS.  opts.K coarsens the time grid to K quantiles of
%            the distinct event times, which is the main cost control; ntree =
%            50 is a good default here.
%
% Output
%   out.times            K-by-1 time grid
%   out.surv_mean        npred-by-K posterior mean survival S(t_k | x)
%   out.haz_mean         npred-by-K posterior mean interval hazard p(t_k | x)
%   out.surv             ndpost-by-(npred*K) draws of survival, subject-major
%                        (row block (i-1)*K + (1:K) is subject i)
%   out.surv_lower/upper 2.5% / 97.5% pointwise credible bands, npred-by-K
%   out.fit              the underlying PBART fit
%   out.x_pred           the covariate matrix predictions refer to
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), JSS 97(1),
% doi:10.18637/jss.v097.i01; and Sparapani et al. (2016), Stat. Med. 35(16).

if nargin < 4, x_test = []; end
if nargin < 5, opts   = struct(); end
opts = bartOptions(opts);

times = times(:);
delta = double(delta(:));
n     = size(x_train, 1);
if numel(times) ~= n || numel(delta) ~= n
    error('survbart:dims', 'times and delta must have %d elements.', n);
end
if any(times <= 0)
    error('survbart:times', 'times must be positive.');
end

% ---- time grid -----------------------------------------------------------
grid = unique(times(delta == 1));
if isempty(grid)
    error('survbart:noevents', 'No events observed.');
end
if ~isempty(opts.K) && opts.K < numel(grid)
    pos  = unique(round(linspace(1, numel(grid), opts.K)));
    grid = grid(pos);
end
grid = grid(:);
K    = numel(grid);

% ---- expand to the long (person-period) data set -------------------------
ki = zeros(n, 1);
ev = delta;
for i = 1:n
    idx = find(grid >= times(i), 1);
    if isempty(idx)              % censored after the last event time
        ki(i) = K;
        ev(i) = 0;
    else
        ki(i) = idx;
    end
end
N     = sum(ki);
tlong = zeros(N, 1);
ylong = zeros(N, 1);
irow  = zeros(N, 1);
pos   = 0;
for i = 1:n
    r        = pos + (1:ki(i));
    tlong(r) = grid(1:ki(i));
    ylong(r) = 0;
    ylong(r(end)) = ev(i);
    irow(r)  = i;
    pos      = pos + ki(i);
end
Xlong = [tlong, x_train(irow, :)];

% ---- prediction grid: every (t_k, x) pair, subject-major -----------------
if isempty(x_test)
    xpred = x_train;
else
    xpred = x_test;
end
npred = size(xpred, 1);
tp    = repmat(grid, npred, 1);
ip    = reshape(repmat(1:npred, K, 1), [], 1);
Xpred = [tp, xpred(ip, :)];

if opts.printevery > 0
    fprintf('survbart: %d subjects, %d time points, %d person-period rows\n', ...
            n, K, N);
end

% ---- fit the probit BART on the long data --------------------------------
sopts           = opts;
sopts.savedraws = true;
fit = pbart(Xlong, ylong, Xpred, sopts);

% ---- survival = cumulative product of (1 - hazard) within each subject ---
nd   = size(fit.prob_test, 1);
haz  = fit.prob_test;                                  % nd-by-(npred*K)
surv = zeros(nd, npred * K);
for i = 1:npred
    cols            = (i - 1) * K + (1:K);
    surv(:, cols)   = cumprod(1 - haz(:, cols), 2);
end

out = struct();
out.type    = 'survbart';
out.times   = grid;
out.K       = K;
out.x_pred  = xpred;
out.fit     = fit;
out.opts    = opts;
out.nlong   = N;

out.haz_mean  = reshape(mean(haz,  1), K, npred)';
out.surv_mean = reshape(mean(surv, 1), K, npred)';
lo = bartPctl(surv, 2.5);
hi = bartPctl(surv, 97.5);
out.surv_lower = reshape(lo, K, npred)';
out.surv_upper = reshape(hi, K, npred)';
if opts.savedraws
    out.surv = surv;
    out.haz  = haz;
else
    out.surv = []; out.haz = [];
    out.fit.prob_test = []; out.fit.prob_train = [];
    out.fit.yhat_test = []; out.fit.yhat_train = [];
end
end
