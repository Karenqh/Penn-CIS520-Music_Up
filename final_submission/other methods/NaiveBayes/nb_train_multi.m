function nb_multi = nb_train_multi(X, Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
Z = bsxfun(@eq,Y,1:10);
nb_multi.p_y = mean(Z);
% temp_sigma = zeros(size(X,2),1);
for i = 1:10
    nb_multi.mu_x_given_y(:,i) = mean(X(Y==i,:));
    nb_multi.sigma_x(:,i) = std(X(Y==i,:));
%     nb_multi.uniq_sigama(:,i) = sum((bsxfun(@minus, nb_multi.mu_x_given_y(:,i), X(Y==i,:)')).^2,2);
%     temp_sigma = temp_sigma +  sum((bsxfun(@minus, nb_multi.mu_x_given_y(:,i), X(Y==i,:)')).^2,2);
end
% nb_multi.sigma_x = sqrt(temp_sigma)/size(Y(Y~=0),1);
% nb_multi.sigma_x(nb_multi.sigma_x == 0) = sqrt(realmin);
% nb_multi.sigma_x = full(nb_multi.sigma_x);
% nb_multi.sigma_x(nb_multi.sigma_x==0) = 0.00001;

end