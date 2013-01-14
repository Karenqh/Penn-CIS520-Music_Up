function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(j,:), X2(i,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.X2_tr = X2'; % DXM
X2_tr = X2'; % DXM
X_tr = X';  % DXN

% Using for loop. Later I may vectorize the loop
for i = 1:m
    K(i,:) = sum(bsxfun(@min, X2_tr(:, i), X_tr));
%     for j = 1:n
%         K(i,j) = sum(min(X2_tr(:, i), X_tr(:, j)));
%     end
end
