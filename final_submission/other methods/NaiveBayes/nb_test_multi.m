function [ log_p_x_and_y] = nb_test_multi( nb_multi, X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
log_p_x_and_y = zeros(size(X,1),10);
for i = 1:size(X,1)
    
%     parfor j = 1:10
%         log_p_x_and_y(i,j) = log(nb_multi.p_y(j)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x.^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,j)).^2./nb_multi.sigma_x.^2)));
%     end

    for j = 1:10
        index = full(nb_multi.sigma_x(:,j)~=0);
        adder = log((1./sqrt(2*pi*nb_multi.sigma_x(index,j).^2)).*exp(-0.5.*((X(i,index)' - nb_multi.mu_x_given_y(index,j)).^2)./(nb_multi.sigma_x(index,j).^2)));
        adder(adder==-inf)=-5;
        log_p_x_and_y(i,j) = log(nb_multi.p_y(j)) +...
            sum(adder);

        
    end
%     log_p_x_and_y(i,1) = log(nb_multi.p_y(1)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(nb_multi.sigma_x ~=0 ,1).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,1)).^2./nb_multi.sigma_x(:,1).^2)));
%     log_p_x_and_y(i,2) = log(nb_multi.p_y(2)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,2).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,2)).^2./nb_multi.sigma_x(:,2).^2)));
%     log_p_x_and_y(i,3) = log(nb_multi.p_y(3)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,3).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,3)).^2./nb_multi.sigma_x(:,3).^2)));
%     log_p_x_and_y(i,4) = log(nb_multi.p_y(4)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,4).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,4)).^2./nb_multi.sigma_x(:,4).^2)));
%     log_p_x_and_y(i,5) = log(nb_multi.p_y(5)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,5).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,5)).^2./nb_multi.sigma_x(:,5).^2)));
%     log_p_x_and_y(i,6) = log(nb_multi.p_y(6)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,6).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,6)).^2./nb_multi.sigma_x(:,6).^2)));
%     log_p_x_and_y(i,7) = log(nb_multi.p_y(7)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,7).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,7)).^2./nb_multi.sigma_x(:,7).^2)));
%     log_p_x_and_y(i,8) = log(nb_multi.p_y(8)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,8).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,8)).^2./nb_multi.sigma_x(:,8).^2)));
%     log_p_x_and_y(i,9) = log(nb_multi.p_y(9)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,9).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,9)).^2./nb_multi.sigma_x(:,9).^2)));
%     log_p_x_and_y(i,10) = log(nb_multi.p_y(10)) + sum(log((1./sqrt(2*pi*nb_multi.sigma_x(:,10).^2)).*exp(-0.5.*(X(i,:)' - nb_multi.mu_x_given_y(:,10)).^2./nb_multi.sigma_x(:,10).^2)));
end

end

