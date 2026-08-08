function im = argmaxrow(P)
[~, im] = max(P, [], 2);
im = im(:);
end
