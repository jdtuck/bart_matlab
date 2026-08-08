function nid = bartAssign(tree, Xb)
%BARTASSIGN Terminal-node index of every row of Xb under TREE.
%
%   nid = bartAssign(tree, Xb)
%
% TREE is a struct with integer arrays var, cut, lc, rc (0 marks a leaf in var,
% child indices in lc/rc).  Xb holds binned covariates from BARTBINX.
% The descent is vectorized: one pass per level of the tree, not per row.

n   = size(Xb, 1);
nid = ones(n, 1);
idx = (1:n)';

while true
    v  = tree.var(nid(idx));
    go = v > 0;
    if ~any(go)
        break;
    end
    idx  = idx(go);
    nd   = nid(idx);
    v    = tree.var(nd);
    k    = tree.cut(nd);
    xv   = Xb(idx + (v - 1) * n);
    left = xv < k;
    nid(idx(left))  = tree.lc(nd(left));
    nid(idx(~left)) = tree.rc(nd(~left));
end
end
