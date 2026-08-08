function p = bartPhi(x)
%BARTPHI Standard normal CDF (no Statistics Toolbox required).
p = 0.5 * erfc(-x / sqrt(2));
end
