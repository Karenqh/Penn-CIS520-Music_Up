function [xval_err info] = train_svm(Xtrain, Ytrain, Xtest, Ytest)

% Scale the feature_vector
Xtrain = bsxfun(@rdivide, Xtrain, sum(Xtrain, 2));
Xtest = bsxfun(@rdivide, Xtest, sum(Xtest, 2));

% Cross validation to get an average error
n_fold = 5;
partitions = make_xval_partition(size(Xtrain, 1), n_fold);
xval_train_set = Xtrain(partitions~=n_fold, :);
xval_train_label = Ytrain(partitions~=n_fold);
xval_test_set = Xtrain(partitions==n_fold, :);
xval_test_label = Ytrain(partitions==n_fold);

% scores = zeros(size(xval_test_set, 10));


% Use kerneled SVM to train. We need probability predictions
kernel = @(x, x2) kernel_intersection(x, x2);

[xval_err info] = kernel_libsvm(xval_train_set, xval_train_label,...
                                xval_test_set, xval_test_label, kernel);

