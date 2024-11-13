function  Wv = Update_W_v(v , V ,Teta , kap, W, gam  )

s = 0; 
Teta = Teta.^kap ; 
nv = length(V);
NC = size(V{v} , 2);
for r=1:nv    
    s = s + Teta(v,r)*V{r}*W{r};
end

S = (V{v}'*s) + (2*(gam*W{v}*eye(NC, NC)));

M = (V{v}'*V{v}*W{v}*sum(Teta(v,:)))+(2*gam*(W{v}*W{v}'*W{v}));

%Wv = W{v}.*((S./M).^(1/4));
Wv = W{v}.*((S./M));
end
