function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);
D = size(X,2);
% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
% XX = repmat(X',m,1);
% XX = bsxfun(@plus, );
% XX2 = reshape(X2', 1, size(X2,1) * size(X2,2))';
% XX2 = repmat(XX2,1,n);

X = X';
for i = 1:n
    K(:,i) = sum(bsxfun(@min, X2', X(:,i)))';
%     K(m,:) = sum(min(XX((i*D+1):(i*(D+1)),:),XX2((i*D+1):(i*(D+1)),:)));
end