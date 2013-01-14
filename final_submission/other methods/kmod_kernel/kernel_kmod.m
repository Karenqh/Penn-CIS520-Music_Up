function K = kernel_kmod( X, X2,gamma, sigma )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a = 1/(exp(gamma/sigma^2)-1);
n = size(X,1);
m = size(X2,1);
K = zeros(m, n);
X = X';
for i = 1:n
   K(:,i) = a * (exp(gamma ./ ((sum(bsxfun(@minus,X2',X(:,i)).^2))' + sigma^2)-1));
end

end

