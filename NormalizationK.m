function [K] = NormalizationK(K)
        n = size(K,2);
            norms = max(1e-15,sqrt(sum(K.^2,1)))';
            K = K*spdiags(norms.^-1,0,n,n);
end