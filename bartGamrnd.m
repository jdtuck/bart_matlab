function x = bartGamrnd(a)
%BARTGAMRND Gamma(a,1) draws, elementwise in the shape vector A.
%
%   x = bartGamrnd(a)
%
% Marsaglia & Tsang (2000) squeeze method, vectorized, with the Johnk/boost
% correction for shapes below one.  Requires no toolbox: chi-square draws are
% chi2(nu) = 2*bartGamrnd(nu/2), and Dirichlet draws are normalized gammas.

sz = size(a);
a  = double(a(:));
if any(a <= 0)
    error('bartGamrnd:shape', 'Shape parameters must be positive.');
end

low = a < 1;
a2  = a + low;                 % work with shape >= 1
d   = a2 - 1/3;
c   = 1 ./ sqrt(9 * d);

n    = numel(a);
x    = zeros(n, 1);
todo = (1:n)';
while ~isempty(todo)
    z  = randn(numel(todo), 1);
    v  = (1 + c(todo) .* z).^3;
    u  = rand(numel(todo), 1);
    ok = (v > 0);
    ok(ok) = log(u(ok)) < 0.5 * z(ok).^2 + d(todo(ok)) .* ...
             (1 - v(ok) + log(v(ok)));
    x(todo(ok)) = d(todo(ok)) .* v(ok);
    todo = todo(~ok);
end

if any(low)
    x(low) = x(low) .* rand(sum(low), 1).^(1 ./ a(low));
end
x = reshape(x, sz);
end
