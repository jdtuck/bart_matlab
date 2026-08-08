function out = bartCore(Xb, ncut, y, w, tau, opts)
%BARTCORE Bayesian backfitting MCMC for a sum-of-trees model.
%
%   out = bartCore(Xb, ncut, y, w, tau, opts)
%
% This is the shared engine behind WBART, PBART, LBART, MBART and SURVBART.
% It targets the posterior of
%
%     y_i = sum_{t=1..m} g(x_i; T_t, M_t) + eps_i,   eps_i ~ N(0, sigma^2/w_i)
%
% with the Chipman, George & McCulloch (2010) priors: a branching-process prior
% on each tree, P(split at depth d) = alpha*(1+d)^(-beta); independent
% N(0, tau^2) leaf means; and (optionally) a scaled inverse chi-square prior on
% sigma^2.  Trees are updated one at a time by Metropolis-Hastings on the
% partial residuals (Bayesian backfitting), with GROW, PRUNE and CHANGE moves.
% Because the leaf prior is conjugate, the leaf means are integrated out of the
% acceptance ratio and then drawn exactly.
%
% Inputs
%   Xb    n-by-p integer cutpoint codes from BARTBINX
%   ncut  1-by-p number of candidate cutpoints per covariate
%   y     n-by-1 response for the trees (any offset already subtracted)
%   w     n-by-1 weights; observation i has variance sigma^2/w(i)
%   tau   leaf prior standard deviation
%   opts  see BARTOPTIONS, plus the engine-only fields
%           sigma        starting value of sigma
%           sampleSigma  draw sigma^2 each sweep (continuous outcomes)
%           lambda       scale of the sigma^2 prior
%           latentfun    [] or @(f) -> [y, w], a data-augmentation step run
%                        before every sweep, given the current sum-of-trees f
%           Xbtest       binned test covariates ([] for none)
%
% Output fields
%   fhat_train / fhat_test  ndpost-by-n draws of the sum-of-trees function
%   sigma                   ndpost-by-1 draws (all ones if not sampled)
%   varcount                ndpost-by-p split counts per covariate
%   trees                   ndpost-by-1 cell of m-by-1 cell of tree structs
%   accept                  GROW/PRUNE/CHANGE acceptance rates
%
% Reference: Sparapani, Spanbauer & McCulloch (2021), "Nonparametric Machine
% Learning and Efficient Computation with Bayesian Additive Regression Trees:
% The BART R Package", JSS 97(1). doi:10.18637/jss.v097.i01

[n, p] = size(Xb);
m      = opts.ntree;
y      = y(:);
w      = w(:);
sigma  = opts.sigma;

nkeep   = opts.ndpost;
totiter = opts.nskip + nkeep * max(1, opts.keepevery);

% ---- state ---------------------------------------------------------------
trees = cell(m, 1);
for j = 1:m
    trees{j} = newTree(8);
end
nid  = ones(n, m);            % leaf index of each observation, per tree
fmat = zeros(n, m);           % contribution of each tree
fhat = zeros(n, 1);

s     = ones(p, 1) / p;       % splitting probabilities
theta = opts.theta;
if theta <= 0, theta = 1; end
rho = opts.rho;
if isempty(rho), rho = p; end

hasTest = isfield(opts, 'Xbtest') && ~isempty(opts.Xbtest);
if hasTest
    ntest = size(opts.Xbtest, 1);
else
    ntest = 0;
end

fhat_train = zeros(nkeep, n);
fhat_test  = zeros(nkeep, ntest);
sigmad     = ones(nkeep, 1);
varcount   = zeros(nkeep, p);
treedraws  = cell(nkeep * double(opts.savetrees), 1);

nprop = zeros(1, 3); nacc = zeros(1, 3);
keep  = 0;

% ---- MCMC ----------------------------------------------------------------
for it = 1:totiter

    % data augmentation (probit latents, Polya-Gamma weights, ...)
    if ~isempty(opts.latentfun)
        [y, w] = opts.latentfun(fhat);
        y = y(:); w = w(:);
    end

    for j = 1:m
        rj = y - fhat + fmat(:, j);                     % partial residual
        [trees{j}, nid(:, j), mv, ac] = ...
            updateTree(trees{j}, nid(:, j), Xb, ncut, rj, w, sigma, tau, opts, s);
        nprop(mv) = nprop(mv) + 1;
        nacc(mv)  = nacc(mv) + ac;
        [trees{j}, fj] = drawMu(trees{j}, nid(:, j), rj, w, sigma, tau);
        fhat = fhat + (fj - fmat(:, j));
        fmat(:, j) = fj;
    end

    if opts.sampleSigma
        sse   = sum(w .* (y - fhat).^2);
        sigma = sqrt((opts.nu * opts.lambda + sse) / ...
                     (2 * bartGamrnd((opts.nu + n) / 2)));
    end

    if opts.sparse
        cnt = splitCounts(trees, p);
        g   = bartGamrnd(theta / p + cnt);
        s   = g / sum(g);
        s   = max(s, 1e-12); s = s / sum(s);
        if opts.theta <= 0
            theta = drawTheta(s, p, rho, opts.a, opts.b);
        end
    end

    if opts.printevery > 0 && mod(it, opts.printevery) == 0
        fprintf('  bartCore: iteration %d of %d (sigma = %.4g)\n', ...
                it, totiter, sigma);
    end

    % ---- store ----------------------------------------------------------
    if it > opts.nskip && mod(it - opts.nskip, max(1, opts.keepevery)) == 0
        keep = keep + 1;
        fhat_train(keep, :) = fhat';
        sigmad(keep)        = sigma;
        varcount(keep, :)   = splitCounts(trees, p)';
        if hasTest || opts.savetrees
            ct = cell(m, 1);
            for j = 1:m
                ct{j} = compactTree(trees{j});
            end
            if opts.savetrees
                treedraws{keep} = ct;
            end
            if hasTest
                ft = zeros(ntest, 1);
                for j = 1:m
                    ft = ft + ct{j}.mu(bartAssign(ct{j}, opts.Xbtest));
                end
                fhat_test(keep, :) = ft';
            end
        end
    end
end

out = struct();
out.fhat_train = fhat_train;
out.fhat_test  = fhat_test;
out.sigma      = sigmad;
out.varcount   = varcount;
out.trees      = treedraws;
out.accept     = nacc ./ max(nprop, 1);
out.nproposed  = nprop;
end

% =========================================================================
% Tree bookkeeping
% =========================================================================
function tree = newTree(K)
% A tree is a flat node table: var = 0 marks a leaf, otherwise (var,cut) is the
% split rule and lc/rc index the children.  Freed slots are recycled.
tree = struct( ...
    'var',    zeros(K, 1), ...
    'cut',    zeros(K, 1), ...
    'lc',     zeros(K, 1), ...
    'rc',     zeros(K, 1), ...
    'parent', zeros(K, 1), ...
    'depth',  zeros(K, 1), ...
    'mu',     zeros(K, 1), ...
    'used',   false(K, 1), ...
    'free',   (2:K)');
tree.used(1) = true;
end

function [tree, idx] = allocNodes(tree, k)
if numel(tree.free) < k
    K    = numel(tree.var);
    newK = max(2 * K, K + k);
    pad  = zeros(newK - K, 1);
    tree.var    = [tree.var;    pad];
    tree.cut    = [tree.cut;    pad];
    tree.lc     = [tree.lc;     pad];
    tree.rc     = [tree.rc;     pad];
    tree.parent = [tree.parent; pad];
    tree.depth  = [tree.depth;  pad];
    tree.mu     = [tree.mu;     pad];
    tree.used   = [tree.used;   false(newK - K, 1)];
    tree.free   = [tree.free;   ((K + 1):newK)'];
end
idx = tree.free(end - k + 1:end);
tree.free(end - k + 1:end) = [];
end

function [tree, cl, cr] = growNode(tree, L, v, k)
[tree, ix] = allocNodes(tree, 2);
cl = ix(1); cr = ix(2);
tree.var(L) = v; tree.cut(L) = k; tree.lc(L) = cl; tree.rc(L) = cr;
tree.var([cl cr])    = 0;
tree.cut([cl cr])    = 0;
tree.lc([cl cr])     = 0;
tree.rc([cl cr])     = 0;
tree.mu([cl cr])     = 0;
tree.parent([cl cr]) = L;
tree.depth([cl cr])  = tree.depth(L) + 1;
tree.used([cl cr])   = true;
end

function tree = pruneNode(tree, N)
cl = tree.lc(N); cr = tree.rc(N);
tree.used([cl cr]) = false;
tree.free = [tree.free; cl; cr];
tree.var(N) = 0; tree.cut(N) = 0; tree.lc(N) = 0; tree.rc(N) = 0;
end

function nog = findNog(tree)
% Internal nodes whose two children are both leaves (candidates for PRUNE and
% for the restricted CHANGE move).
in = find(tree.used & tree.var > 0);
if isempty(in)
    nog = zeros(0, 1);
    return;
end
nog = in(tree.var(tree.lc(in)) == 0 & tree.var(tree.rc(in)) == 0);
end

function ct = compactTree(tree)
K = find(tree.used, 1, 'last');
ct = struct('var', tree.var(1:K), 'cut', tree.cut(1:K), ...
            'lc',  tree.lc(1:K),  'rc',  tree.rc(1:K), ...
            'mu',  tree.mu(1:K));
end

function cnt = splitCounts(trees, p)
cnt = zeros(p, 1);
for j = 1:numel(trees)
    t  = trees{j};
    vv = t.var(t.used & t.var > 0);
    if ~isempty(vv)
        cnt = cnt + accumarray(vv, 1, [p 1]);
    end
end
end

% =========================================================================
% One Metropolis-Hastings move on a single tree
% =========================================================================
function [tree, nid, move, acc] = updateTree(tree, nid, Xb, ncut, r, w, sigma, tau, opts, s)
n      = size(Xb, 1);
leaves = find(tree.used & tree.var == 0);
nl     = numel(leaves);
nog    = findNog(tree);
acc    = 0;

if nl == 1
    move = 1; pgF = 1;              % only GROW is possible from a stump
else
    u = rand;
    if u < opts.pg
        move = 1; pgF = opts.pg;
    elseif u < opts.pg + opts.pp
        move = 2; pgF = opts.pg;
    else
        move = 3; pgF = opts.pg;
    end
end

switch move

    % ------------------------------------------------------------------ GROW
    case 1
        L   = leaves(randi(nl));
        obs = find(nid == L);
        if isempty(obs), return; end
        [v, k] = proposeRule(ncut, s, opts);
        gl = Xb(obs + (v - 1) * n) < k;
        nL = sum(gl);
        if nL == 0 || nL == numel(obs), return; end     % empty child: reject
        il = obs(gl); ir = obs(~gl);
        [Wl, Sl] = suff(w, r, il);
        [Wr, Sr] = suff(w, r, ir);
        llik = logML(Wl, Sl, sigma, tau) + logML(Wr, Sr, sigma, tau) ...
             - logML(Wl + Wr, Sl + Sr, sigma, tau);

        d   = tree.depth(L);
        pd  = opts.alpha * (1 + d)^(-opts.beta);
        pc  = opts.alpha * (2 + d)^(-opts.beta);
        lpr = log(pd) + 2 * log1p(-pc) - log1p(-pd);

        nogNew = numel(nog) + 1 - double(siblingIsLeaf(tree, L));
        ltr    = log(opts.pp) - log(nogNew) - (log(pgF) - log(nl));

        if log(rand) < llik + lpr + ltr
            [tree, cl, cr] = growNode(tree, L, v, k);
            nid(il) = cl; nid(ir) = cr;
            acc = 1;
        end

    % ----------------------------------------------------------------- PRUNE
    case 2
        if isempty(nog), return; end
        N  = nog(randi(numel(nog)));
        cl = tree.lc(N); cr = tree.rc(N);
        il = find(nid == cl); ir = find(nid == cr);
        [Wl, Sl] = suff(w, r, il);
        [Wr, Sr] = suff(w, r, ir);
        llik = logML(Wl + Wr, Sl + Sr, sigma, tau) ...
             - logML(Wl, Sl, sigma, tau) - logML(Wr, Sr, sigma, tau);

        d   = tree.depth(N);
        pd  = opts.alpha * (1 + d)^(-opts.beta);
        pc  = opts.alpha * (2 + d)^(-opts.beta);
        lpr = -(log(pd) + 2 * log1p(-pc) - log1p(-pd));

        nlNew = nl - 1;
        if nlNew == 1, pgR = 1; else pgR = opts.pg; end
        ltr = log(pgR) - log(nlNew) - (log(opts.pp) - log(numel(nog)));

        if log(rand) < llik + lpr + ltr
            tree = pruneNode(tree, N);
            nid(il) = N; nid(ir) = N;
            acc = 1;
        end

    % ---------------------------------------------------------------- CHANGE
    case 3
        if isempty(nog), return; end
        N   = nog(randi(numel(nog)));
        cl  = tree.lc(N); cr = tree.rc(N);
        il0 = find(nid == cl); ir0 = find(nid == cr);
        obs = [il0; ir0];
        if isempty(obs), return; end
        [v, k] = proposeRule(ncut, s, opts);
        gl = Xb(obs + (v - 1) * n) < k;
        nL = sum(gl);
        if nL == 0 || nL == numel(obs), return; end
        il = obs(gl); ir = obs(~gl);
        [Wl0, Sl0] = suff(w, r, il0);
        [Wr0, Sr0] = suff(w, r, ir0);
        [Wl,  Sl ] = suff(w, r, il);
        [Wr,  Sr ] = suff(w, r, ir);
        llik = logML(Wl,  Sl,  sigma, tau) + logML(Wr,  Sr,  sigma, tau) ...
             - logML(Wl0, Sl0, sigma, tau) - logML(Wr0, Sr0, sigma, tau);
        % Tree topology is unchanged, and the rule proposal equals the rule
        % prior, so the structure prior and transition terms cancel.
        if log(rand) < llik
            tree.var(N) = v; tree.cut(N) = k;
            nid(il) = cl; nid(ir) = cr;
            acc = 1;
        end
end
end

function tf = siblingIsLeaf(tree, node)
par = tree.parent(node);
if par == 0
    tf = false;
    return;
end
sib = tree.lc(par);
if sib == node, sib = tree.rc(par); end
tf = (tree.var(sib) == 0);
end

function [W, S] = suff(w, r, idx)
wi = w(idx);
W  = sum(wi);
S  = sum(wi .* r(idx));
end

function v = logML(W, S, sigma, tau)
% Log marginal likelihood of a node's data with the leaf mean integrated out,
% dropping constants common to all tree structures.
s2  = sigma^2;
t2  = tau^2;
den = s2 + t2 * W;
v   = 0.5 * log(s2 / den) + t2 * S^2 / (2 * s2 * den);
end

function [v, k] = proposeRule(ncut, s, opts)
if opts.sparse
    v = find(rand <= cumsum(s), 1);
    if isempty(v), v = numel(ncut); end
else
    v = randi(numel(ncut));
end
k = randi(ncut(v));
end

% =========================================================================
% Conjugate leaf-mean draw
% =========================================================================
function [tree, f] = drawMu(tree, nid, r, w, sigma, tau)
K   = numel(tree.var);
Sv  = accumarray(nid, w .* r, [K 1]);
Wv  = accumarray(nid, w,      [K 1]);
lf  = find(tree.used & tree.var == 0);
A   = Wv(lf) / sigma^2 + 1 / tau^2;          % posterior precision
tree.mu(lf) = (Sv(lf) / sigma^2) ./ A + randn(numel(lf), 1) ./ sqrt(A);
f = tree.mu(nid);
end

% =========================================================================
% Linero (2018) sparse Dirichlet prior: draw the concentration parameter
% =========================================================================
function theta = drawTheta(s, p, rho, a, b)
L    = 1000;
lam  = (1:L)' / (L + 1);
th   = rho * lam ./ (1 - lam);
sl   = sum(log(s));
lp   = gammaln(th) - p * gammaln(th / p) + (th / p - 1) * sl ...
     + (a - 1) * log(lam) + (b - 1) * log1p(-lam);
lp   = lp - max(lp);
pr   = exp(lp); pr = pr / sum(pr);
theta = th(find(rand <= cumsum(pr), 1));
if isempty(theta), theta = th(end); end
end
