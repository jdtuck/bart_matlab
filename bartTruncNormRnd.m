function z = bartTruncNormRnd(mu, pos)
%BARTTRUNCNORMRND Draw from N(mu,1) truncated to one side of zero.
%
%   z = bartTruncNormRnd(mu, pos)
%
% pos(i) true  -> z(i) ~ N(mu(i),1) restricted to (0, Inf)
% pos(i) false -> z(i) ~ N(mu(i),1) restricted to (-Inf, 0)
%
% This is the Albert & Chib (1993) latent draw for probit BART.  Naive
% rejection is used when the truncation point is not far in the tail, and
% Robert's (1995) exponential-proposal rejection otherwise, so the sampler
% stays exact and fast even when |mu| is large (probabilities near 0 or 1).

mu  = mu(:);
pos = logical(pos(:));
n   = numel(mu);

% Reduce to the positive-truncation case by reflection.
s = ones(n, 1);
s(~pos) = -1;
m = s .* mu;          % draw x ~ N(m,1) on (0,Inf), then z = s*x
t = -m;               % standardized lower bound for the N(0,1) part

x = zeros(n, 1);

% --- Region 1: bound not deep in the tail -> plain rejection sampling ------
todo = find(t < 0.45);
while ~isempty(todo)
    cand = randn(numel(todo), 1);
    ok   = cand > t(todo);
    x(todo(ok)) = cand(ok);
    todo = todo(~ok);
end

% --- Region 2: far tail -> Robert (1995) exponential proposal --------------
todo = find(t >= 0.45);
if ~isempty(todo)
    al = (t + sqrt(t.^2 + 4)) / 2;
    while ~isempty(todo)
        e    = -log(rand(numel(todo), 1));
        cand = t(todo) + e ./ al(todo);
        u    = rand(numel(todo), 1);
        ok   = u <= exp(-0.5 * (cand - al(todo)).^2);
        x(todo(ok)) = cand(ok);
        todo = todo(~ok);
    end
end

z = s .* (m + x);
end
