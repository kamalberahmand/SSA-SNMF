function  Vf = Update_V_v(NC,alph, kap, v, S_tilde , V , W, Teta)

nv = length(V);
s = 0;
n = size(V{v} ,1);
Teta = Teta.^kap;
for r =1:nv    
    s = s+ Teta(v,r)*V{r}*W{v}*W{r}';
end

Ss = (2*S_tilde*V{v}) + s + (NC*alph*ones(n , NC));  
M = (2*(V{v}*V{v}')*V{v}) + (V{v}*(W{v}*W{v}')*sum(Teta(v,:))) + (NC*alph*V{v}*ones(NC, NC));

%Vf = V{v}.*((Ss./M).^(1/4));
Vf = V{v}.*((Ss./M));
end
