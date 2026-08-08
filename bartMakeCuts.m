function cuts = bartMakeCuts(X, numcut, usequants)
%BARTMAKECUTS Candidate split points for each column of X.
%
%   cuts = bartMakeCuts(X, numcut, usequants)
%
% Returns a 1-by-p cell array; cuts{j} is a sorted column vector of candidate
% thresholds for covariate j.  A split rule is "x(:,j) <= cuts{j}(c)".
%
% If a covariate takes few distinct values (<= numcut+1) the cutpoints are the
% midpoints between consecutive distinct values, which gives every possible
% split exactly once (this covers binary dummy variables automatically).
% Otherwise the grid is either equally spaced on the range (usequants = false,
% the BART package default) or equally spaced in probability (usequants = true).

if nargin < 2 || isempty(numcut),    numcut = 100;    end
if nargin < 3 || isempty(usequants), usequants = false; end

p = size(X, 2);
cuts = cell(1, p);

for j = 1:p
    xj = X(:, j);
    xj = xj(isfinite(xj));
    u  = unique(xj);
    if numel(u) < 2
        % Degenerate covariate: keep one (unusable) cutpoint so indexing works.
        cuts{j} = u(1);
        continue;
    end
    if numel(u) - 1 <= numcut
        cuts{j} = (u(1:end-1) + u(2:end)) / 2;
    elseif usequants
        pr = (1:numcut)' / (numcut + 1);
        cuts{j} = unique(bartQuantile(xj, pr));
    else
        a = min(u); b = max(u);
        cuts{j} = a + (b - a) * ((1:numcut)' / (numcut + 1));
    end
    cuts{j} = cuts{j}(:);
end
end

function q = bartQuantile(x, pr)
% Linear-interpolation quantiles (no Statistics Toolbox required).
x = sort(x(:));
n = numel(x);
h = (n - 1) * pr(:) + 1;
lo = floor(h); hi = ceil(h);
lo = min(max(lo, 1), n); hi = min(max(hi, 1), n);
q = x(lo) + (h - lo) .* (x(hi) - x(lo));
end
