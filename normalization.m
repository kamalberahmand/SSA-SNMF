function V = normalization(V)
V = V./repmat(sum(V')' ,1, size(V,2));
end