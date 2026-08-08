function out = mbart(x_train, y_train, x_test, opts)
%MBART BART for a categorical outcome with J >= 2 unordered levels.
%
%   out = mbart(x_train, y_train)
%   out = mbart(x_train, y_train, x_test, opts)
%
% The multinomial likelihood is factored into a sequence of conditional
% binary problems (the conditional-probability decomposition used by the BART
% package's mbart):
%
%   P(Y = 1)            = p_1(x)
%   P(Y = j | Y >= j)   = p_j(x),      j = 1, ..., J-1
%   P(Y = j)            = p_j(x) * prod_{l < j} (1 - p_l(x))
%   P(Y = J)            = prod_{l < J} (1 - p_l(x))
%
% Because the likelihood factorizes across j, each p_j is fitted by an
% independent probit BART (PBART) on the subsample still "at risk", i.e. the
% rows with y >= j.  Each of those fits predicts at every training and test
% point, and the draws are then multiplied together draw by draw, so the output
% is a proper joint posterior sample of the category probabilities.
%
% Inputs
%   y_train  n-by-1 category labels: numeric, logical, char/cellstr or
%            categorical.  Levels are ordered by UNIQUE; the last level is the
%            reference category J.
%
% Output
%   out.levels             J-by-1 list of levels in the order used
%   out.prob_train_mean    n-by-J posterior mean P(Y = j | x_i)
%   out.prob_test_mean     ntest-by-J posterior mean at x_test
%   out.prob_train         ndpost-by-n-by-J draws (if opts.savedraws)
%   out.prob_test          ndpost-by-ntest-by-J draws (if opts.savedraws)
%   out.pred_train         n-by-1 most probable level for each training row
%   out.pred_test          ntest-by-1 most probable level at x_test
%   out.fits               1-by-(J-1) cell of the underlying PBART fits
%   out.varprob            average split proportions across the J-1 fits
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), JSS 97(1).
% doi:10.18637/jss.v097.i01

if nargin < 3, x_test = []; end
if nargin < 4, opts   = struct(); end
opts = bartOptions(opts);

n = size(x_train, 1);
if iscell(y_train) || ischar(y_train)
    [levels, ~, yi] = unique(cellstr(y_train));
else
    [levels, ~, yi] = unique(y_train(:));
end
yi = yi(:);
J  = numel(levels);
if J < 2
    error('mbart:levels', 'y_train must have at least two levels.');
end
if J == 2
    warning('mbart:binary', 'Only two levels found; PBART is the direct choice.');
end

ntest = size(x_test, 1);
xpred = [x_train; x_test];               % every fit predicts everywhere

fits = cell(1, J - 1);
nd   = opts.ndpost;
cp   = zeros(nd, n + ntest, J - 1);      % conditional probabilities p_j

sopts           = opts;
sopts.savedraws = true;                  % needed to combine the fits

for j = 1:(J - 1)
    atrisk = (yi >= j);
    yj     = double(yi(atrisk) == j);
    if all(yj == 1) || all(yj == 0)
        % Degenerate conditional: no information, use the empirical value.
        cp(:, :, j) = mean(yj);
        fits{j} = struct('degenerate', true, 'p', mean(yj));
        continue;
    end
    if opts.printevery > 0
        fprintf('mbart: fitting conditional model %d of %d (n = %d)\n', ...
                j, J - 1, sum(atrisk));
    end
    fj = pbart(x_train(atrisk, :), yj, xpred, sopts);
    cp(:, :, j) = fj.prob_test;
    fj.prob_test = [];                   % keep the stored fit small
    fj.yhat_test = [];
    fits{j} = fj;
end

% ---- assemble the category probabilities draw by draw --------------------
prob  = zeros(nd, n + ntest, J);
carry = ones(nd, n + ntest);             % prod_{l < j} (1 - p_l)
for j = 1:(J - 1)
    prob(:, :, j) = cp(:, :, j) .* carry;
    carry         = carry .* (1 - cp(:, :, j));
end
prob(:, :, J) = carry;

pm = reshape(mean(prob, 1), [n + ntest, J]);

out = struct();
out.type   = 'mbart';
out.levels = levels;
out.fits   = fits;
out.opts   = opts;

out.prob_train_mean = pm(1:n, :);
[~, im]             = max(out.prob_train_mean, [], 2);
out.pred_train      = levels(im);
if ntest > 0
    out.prob_test_mean = pm(n + 1:end, :);
    [~, im]            = max(out.prob_test_mean, [], 2);
    out.pred_test      = levels(im);
else
    out.prob_test_mean = [];
    out.pred_test      = [];
end
if opts.savedraws
    out.prob_train = prob(:, 1:n, :);
    if ntest > 0
        out.prob_test = prob(:, n + 1:end, :);
    else
        out.prob_test = [];
    end
else
    out.prob_train = []; out.prob_test = [];
end

vp = [];
for j = 1:(J - 1)
    if isfield(fits{j}, 'varprob')
        vp = [vp; fits{j}.varprob];      %#ok<AGROW>
    end
end
out.varprob = mean(vp, 1);
end
