function M = Create_KNNgraph(k, data)

%sigma = 1/(sum(var(data))); % Gaussian kernel

M = zeros(size(data ,2) , size(data,2));
for i=1:size(data ,2)
   d1 = sum((repmat(data(:,i) ,1,size(data ,2)) - data).^2);
   d1(isnan(d1)) = 0;
   sigma = sum(d1);
   dist1 = sqrt(d1); 
   %sigma = sum(dist1);

   [~ ,c] = sort(dist1);
   M(i  , c(1,1:k+1)) = exp(-d1(1,c(1,1:k+1))./(sigma.^2)); 
end


% for i=1:size(data ,2)
%      d1 = sum((repmat(data(:,i) ,1,size(data ,2)) - data).^2);
%      dist1 = sqrt(d1); 
%      [~ ,c] = sort(dist1);
%      x = data(:,i)'*data./(sqrt(sum(data(:,i).^2))*(sqrt(sum(data.^2))));
%      M(i  , c(1,1:k+1)) =x(1 ,c(1,1:k+1)) ; 
% end

M = (M+M')./2;
end