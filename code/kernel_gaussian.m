function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over
% columns. Sparse column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
X2_tr = transpose(X2); % DXM
X_tr = transpose(X);  % DXN

% Using for loop. Later I may vectorize the loop
for i = 1:m
    for j = 1:n
        K(i,j) = exp(-sum((X2_tr(:, i) - X_tr(:, j)).^2)/2/(sigma^2));
    end
    
end

% % After you've computed K, make sure not a sparse matrix anymore
% K = full(K);
