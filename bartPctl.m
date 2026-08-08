function q = bartPctl(A, pr)
%BARTPCTL Empirical percentile of posterior draws (no toolbox required).
%
%   q = bartPctl(v, 2.5)    % v a vector  -> scalar
%   q = bartPctl(A, 97.5)   % A ndraw-by-k -> 1-by-k, one value per column
%
% Uses the nearest-rank definition, which is the convention the BART package
% uses for credible intervals of posterior draws.

if isvector(A)
    A = A(:);
end
A = sort(A, 1);
n = size(A, 1);
h = round(pr / 100 * n + 0.5);
h = max(1, min(n, h));
q = A(h, :);
end
