function pca_on_feature(X)
% Scaling
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, max(abs(X)));

[coeff, score, latent] = princomp(X);

cumsum(latent)./sum(latent)