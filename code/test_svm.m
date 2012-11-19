function [test_err ranks info] = test_svm(Xtrain, Ytrain, Xtest, Ytest)
% Scale the feature_vector
Xtrain = bsxfun(@rdivide, Xtrain, sum(Xtrain, 2));
Xtest = bsxfun(@rdivide, Xtest, sum(Xtest, 2));

% Compute kernel matrices for training and testing.
K = kernel_intersection(Xtrain, Xtrain);
Ktest = kernel_intersection(Xtrain, Xtest);
display('Kernel is done!')

model = svmtrain(Ytrain, [(1:size(K,1))' K], sprintf('-t 4 -c %g -b 1', 100));
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model, '-b 1');
test_err = mean(yhat~=Ytest);

scores = zeros(size(Xtest,1), 10);
label_seq = (model.Label)';
for i = 1:size(Xtest,1)
    scores(i, label_seq) = vals(i, :);
end
ranks = get_ranks(scores);
loss = rank_loss(ranks, Ytest)


% Optionally we can look at more information from training/testing.
info.vals = vals;
info.yhat = yhat;
info.model = model;

