function om = bartPolyaGammaRnd(c, K)
%BARTPOLYAGAMMARND Draws from the Polya-Gamma distribution PG(1,c).
%
%   om = bartPolyaGammaRnd(c)      % K = 100 terms
%   om = bartPolyaGammaRnd(c, K)
%
% Uses the infinite-sum representation of Polson, Scott & Windle (2013),
%
%   PG(1,c) = (1/(2*pi^2)) * sum_{j=1..Inf} g_j / ((j-1/2)^2 + c^2/(4*pi^2)),
%   g_j ~ Exp(1) independent,
%
% truncated after K terms with the (deterministic) mean of the remaining tail
% added back.  The residual error is O(K^-2) in the mean and is negligible for
% K >= 100; this is an approximate sampler, unlike the exact Devroye method
% used inside the R BART package, so LBART is best treated as a convenience
% alternative to the exact probit link in PBART.

if nargin < 2 || isempty(K), K = 100; end

c  = abs(c(:));
n  = numel(c);
j  = (1:K);
den = bsxfun(@plus, (j - 0.5).^2, c.^2 / (4 * pi^2));     % n-by-K
g   = -log(rand(n, K));                                    % Exp(1)
om  = sum(g ./ den, 2) / (2 * pi^2);

% Mean of the discarded tail, sum_{j>K} 1/((j-1/2)^2 + a) ~= 1/K for large K.
tail = 1 ./ (2 * pi^2 * (K + 0.5 + c.^2 / (4 * pi^2 * (K + 0.5))));
om   = om + tail;

om = max(om, 1e-10);
end
