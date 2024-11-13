function Teta = Update_Teta(et , kap, V, W)
nv = length(V);
n = size(V{1} ,1);
Teta = zeros(nv);
for v=1:nv
    for r=1:nv
       Teta(v,r) = (et/(kap* norm(((V{v}*W{v}) - (V{r}*W{r})),'fro')))^(1/(kap -1)); 
    end
end

end