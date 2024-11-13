clc; 
%% input data
Y = gt ;
NC = size(unique(gt) ,1);
nv = length(fea);

%% data normalization
for v=1:nv
    data{v} = fea{v}';
    data{v} = data{v}./repmat(sqrt(sum(data{v}.^2)), size(data{v},1) , 1);       
end

%% parameters
maxitr = 500;
eps = 0.001;
kk = [5, 7, 10, 15]; % k nearest number
kap = 0.95;
beta = 0.1; % B|S-Z|
lam = 0.2;  % manifold

% constraints
alpha = 1; % that is for V1 =1 constraint
gam = 1 ;   % that is for WW'=I constraint
et = 1;
perlab = 0.3;
k = kk(3); %
% experiments 

%% creating the similarity matrix M 
D_M = cell(1,nv);     
for v=1:nv
    M{v} =                                                                      Create_KNNgraph(k, data{v});
    M{v}(isnan(M{v}))=0;
    D_M{v} = diag(sum(M{v}));                  
                                                                                                                                                                                                      end
 
%% generated results 
Prec = [];
Re =[]; 
Pur =[];                                                                                                                                                                                                     
                                                                                                                                                                               
F = [];                                                   
ARI =[];         
nn =[];                                                       
En =[];                                                             
AC_av =[];                                                                         
TT =[];

perlab= 0.5;
%for perlab=0:0.1:0.5

%% loop of main parameters
%kap = 0.95;
%beta = 0.5; % B|S-Z|
%lam = 0.5;  % manifold

%  for beta=0.1:0.1:1
%      for lam=0.1:0.1:1
%          for kap=0.7:0.1:0.9
n = size(Y ,1);

% %  midle results
Fscore_T = [];
Precision_T = []; 
Recall_T = [];
nmi_T = [];
AR_T = []; 
Entropy_T = [];
ACC_T = [];
Purity_T = [];
timess = [];
%for xx=1:5

%% generating superviosory information

[indices,count]=SelLabSam_Semi_2(Y,perlab);
ind=[1:1:n];
ind(indices)=[];
testlabel=Y(ind);
trainlabel = Y(indices);

%% construct the pairwise constraint matrix Z
Z = zeros(n);
cannot_links = [];
for i = 1:count
    a = indices(i);    %%% location of labeled data point
    for j = i+1:count
        b = indices(j);
        if Y(a)==Y(b)
           Z(a,b)=1;
        else
           [cannot_links] = [cannot_links
               [a b]];
        end
    end
end
Z = (Z+Z');
Z = Z + eye(n);

%% initialization 
S = cell(1,nv);
S_tilde = cell(1,nv);
V = cell(1,nv);
for v=1:nv    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S
    Ss = rand(n , n);
    Ss = NormalizationK(Ss);
    % must links
    S{v} = max(Ss , Z);
    % cannot links
    if size(cannot_links,1)>0
    S{v}(cannot_links(:,1)',cannot_links(:,2)') = 0;
    S{v}(cannot_links(:,2)',cannot_links(:,1)') = 0;
    end
    S_tilde{v} = (S{v}+S{v}')./2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V
    V{v} = rand(n, NC);
    V{v} = normalization(V{v});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% W
    W{v} = eye(NC);
end
Teta = rand(nv, nv);
%Teta = normalization(Teta);
Teta(:,:) = 1/nv;

%% iterative procses
e = [] ;
%Of_old= Cost_function(kap, S_tilde , V, beta, S, Z, lam, L_M, Teta, W);
Of_old = inf;
for itr=1:maxitr 
    
    for v=1:nv         
        % update S{v} 
        S{v} = S{v}.*((((2*V{v}*V{v}') + (2*beta*Z)+ (lam*M{v}))./(S{v}' + ((1+2*beta)*S{v}) + (lam*D_M{v}))));
        S{v}(isnan(S{v})) =0 ;
        S{v}(isinf(S{v})) =0 ;
        
        % set suprvisory information 
        %S{v} = max(S{v} , Z);
        %S{v}(cannot_links(:,1)',cannot_links(:,2)') = 0;
        %S{v}(cannot_links(:,2)',cannot_links(:,1)') = 0;
        
        S_tilde{v} = (S{v}+S{v}')./2;
        S_tilde{v}(isnan(S_tilde{v})) =0 ;
        S_tilde{v}(isinf(S_tilde{v})) =0 ;
      
        
        % update V{v}
        V{v} = Update_V_v(NC, alpha, kap, v, S_tilde{v}, V , W, Teta);
        %V{v} = normalization(V{v});
        V{v}(isnan(V{v})) = 0;
        V{v}(isinf(V{v}))=0;

        
        % Update W{v}
        W{v} = Update_W_v(v , V ,Teta , kap, W, gam);
        W{v}(isnan(W{v})) = 0;
        W{v}(isinf(W{v})) = 0;

        
    end
    Teta = Update_Teta(et , kap, V, W);
    Teta(isnan(Teta))=0;
    Teta(isinf(Teta))=0;    
    %Teta = normalization(Teta);

   % Objective function 
   Of_new = Cost_function(kap, S_tilde , V, beta, S, Z, lam, L_M, Teta, W) ;
   
   if abs(Of_old - Of_new) <= eps
       break;       
   end
   e = [e abs(Of_old - Of_new)];
   Of_old = Of_new;
end

V_star = 0 ; 
for v=1:nv 
    V_star = V_star + V{v}*W{v};
end

V_star = V_star/nv; 

%% evaluation 
% [~ ,predY] = max(V_star'); 
% 
% [Fscore, Precision, Recall, nmi, AR, Entropy, ACC, Purity] = Clustering8Measure(Y, predY');
% 
% [Fscore_T] = [Fscore_T Fscore];
% [Precision_T] = [Precision_T Precision]; 
% [Recall_T] = [Recall_T Recall];
% [nmi_T] = [nmi_T nmi];
% [AR_T] = [AR_T AR]; 
% [Entropy_T] = [Entropy_T Entropy];
% [ACC_T] = [ACC_T ACC];
% [Purity_T] = [Purity_T Purity];
% [timess] = [timess itr]; 
% end % xx

% [Prec] = [Prec sum(Precision_T)/size(Precision_T ,2)];
% [Re] =[Re sum(Recall_T)/size(Recall_T ,2)];
% [Pur] =[Pur sum(Purity_T)/size(Purity_T ,2)];
% [F] = [F sum(Fscore_T)/size(Fscore_T ,2)];
% [ARI] =[ARI sum(AR_T)/size(AR_T , 2)];
% %[nn] =[nn sum(nmi_T)/size(nmi_T , 2)];
% [nn] =[nn max(nmi_T)];
% [En] =[En sum(Entropy_T)/size(Entropy_T , 2)];
% [AC_av] =[AC_av sum(ACC_T)/size(ACC_T ,2)]; 
% [TT] =[TT sum(timess)/size(timess ,2)];

% 
%     end % kap
%   end % lam
% end % beta
% 
% end
    
    
    
    
    
    
    
    
    