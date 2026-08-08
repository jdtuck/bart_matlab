function Xb = bartBinX(X, cuts)
%BARTBINX Encode covariates as integer positions in the cutpoint grid.
%
%   Xb = bartBinX(X, cuts)
%
% Xb(i,j) = #{c : cuts{j}(c) < X(i,j)}, so the split rule
%   "X(i,j) <= cuts{j}(c)"   is equivalent to   "Xb(i,j) < c".
% All tree traversal is then integer comparison, and new data are binned with
% the training cutpoints so that fitted trees transfer exactly.

[n, p] = size(X);
if numel(cuts) ~= p
    error('bartBinX:dims', 'X has %d columns but cuts has %d entries.', p, numel(cuts));
end
Xb = zeros(n, p);
for j = 1:p
    cj = cuts{j}(:)';
    if isempty(cj)
        continue;
    end
    Xb(:, j) = sum(bsxfun(@lt, cj, X(:, j)), 2);
end
end
