function o = bartOptions(varargin)
%BARTOPTIONS Default options for the BART samplers, with optional overrides.
%
%   o = bartOptions()                      % all defaults
%   o = bartOptions('ntree',50,'ndpost',500)
%   o = bartOptions(opts)                  % fill in missing fields of OPTS
%   o = bartOptions(opts,'ntree',50)       % struct + overrides
%
% Fields
%   MCMC
%     ntree      number of trees in the sum-of-trees (default 200; 50 is a
%                good choice for pbart/lbart/survbart, per Sparapani et al.)
%     ndpost     number of posterior draws kept              (1000)
%     nskip      burn-in iterations discarded                (100)
%     keepevery  thinning: keep every k-th draw after burn-in (1)
%     printevery print progress every k iterations, 0 = quiet (100)
%     seed       RNG seed; [] leaves the stream untouched    ([])
%
%   Cutpoints (splitting grid)
%     numcut     max number of candidate cutpoints per covariate (100)
%     usequants  true -> quantile grid, false -> uniform grid (false)
%
%   Tree / leaf priors  (Chipman, George & McCulloch 2010)
%     alpha,beta  P(node at depth d splits) = alpha*(1+d)^(-beta) (0.95, 2)
%     k           leaf shrinkage: bigger k = more shrinkage       (2)
%
%   Error variance prior (continuous outcomes only)
%     nu, q       sigma^2 ~ nu*lambda/chi^2_nu with
%                 P(sigma < sigest) = q                           (3, 0.90)
%     sigest      data-based scale for the prior; [] = estimate   ([])
%     w           observation weights: var(y_i) = sigma^2/w_i     ([])
%
%   Sparse (Dirichlet) variable-selection prior (Linero 2018)
%     sparse      use Dirichlet splitting probabilities        (false)
%     theta       Dirichlet concentration; 0 -> sample it       (0)
%     a,b,rho     hyperparameters for the theta prior          (0.5,1,[]=p)
%
%   Proposal probabilities for the tree Metropolis-Hastings step
%     pg,pp,pc    GROW / PRUNE / CHANGE probabilities  (0.35,0.35,0.30)
%
%   Output control
%     savetrees   keep the sampled trees (needed by bartPredict) (true)
%     savedraws   keep full draw-by-draw fits, not just means    (true)
%
%   Outcome-specific
%     pgterms     terms in the Polya-Gamma series used by lbart  (100)
%     K           number of time points in the survbart grid;
%                 [] = every distinct event time                 ([])
%
% See also WBART, PBART, LBART, MBART, SURVBART, BARTPREDICT

o = struct( ...
    'ntree',     200, ...
    'ndpost',    1000, ...
    'nskip',     100, ...
    'keepevery', 1, ...
    'printevery',100, ...
    'seed',      [], ...
    'numcut',    100, ...
    'usequants', false, ...
    'alpha',     0.95, ...
    'beta',      2, ...
    'k',         2, ...
    'nu',        3, ...
    'q',         0.90, ...
    'sigest',    [], ...
    'w',         [], ...
    'sparse',    false, ...
    'theta',     0, ...
    'a',         0.5, ...
    'b',         1, ...
    'rho',       [], ...
    'pg',        0.35, ...
    'pp',        0.35, ...
    'pc',        0.30, ...
    'savetrees', true, ...
    'savedraws', true, ...
    'pgterms',   100, ...
    'K',         []);

args = varargin;
if ~isempty(args) && isstruct(args{1})
    user = args{1};
    args = args(2:end);
else
    user = struct();
end
if mod(numel(args), 2) ~= 0
    error('bartOptions:pairs', 'Name/value arguments must come in pairs.');
end
for i = 1:2:numel(args)
    user.(args{i}) = args{i+1};
end

fn = fieldnames(user);
for i = 1:numel(fn)
    if ~isfield(o, fn{i})
        warning('bartOptions:unknown', 'Ignoring unknown option "%s".', fn{i});
        continue;
    end
    o.(fn{i}) = user.(fn{i});
end

if ~isempty(o.seed)
    rand('state', o.seed);   %#ok<RAND>  (works in MATLAB and Octave)
    randn('state', o.seed);  %#ok<RAND>
end
end
