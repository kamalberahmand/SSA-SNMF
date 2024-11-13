function of = Cost_function(kap, S_tilde , V, beta, S, Z, lam, L_M, Teta, W)

nv = length(V); 
of = 0 ;
for v=1:nv     
    of = of + norm((S_tilde{v} - (V{v}*V{v}')), 'fro')^2 + (beta*norm((S{v}-Z), 'fro')^2) + (lam*sum(diag(L_M{v}*S{v}))) ;            
    for r=1:nv
        
        of = of + (Teta(v,r)^kap)* (norm(((V{v}*W{v}) - (V{r}*W{r})),'fro')^2);              
    end
end

end