function x = bartPhiInv(p)
%BARTPHIINV Standard normal quantile function (no Statistics Toolbox required).
x = -sqrt(2) * erfcinv(2 * p);
end
