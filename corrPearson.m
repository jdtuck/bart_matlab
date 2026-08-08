function r = corrPearson(a, b)
%CORRPEARSON Pearson correlation of two vectors (no Statistics Toolbox needed).
a = a(:) - mean(a(:));
b = b(:) - mean(b(:));
r = (a' * b) / sqrt((a' * a) * (b' * b));
end
