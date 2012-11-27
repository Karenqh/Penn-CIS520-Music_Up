function [info_4] = train_svm_4_out(Xtrain, Ytrain)


% Pick out genre 4
Y_4 = ones(length(Ytrain),1);
Y_4(Ytrain~=4) = [];  
X_4 = Xtrain;
X_4(Ytrain~=4, :) = [];

Y_45 = Ytrain;
Y_45(~bsxfun(@or, Ytrain==4, Ytrain==5)) = [];
X_45 = Xtrain;
X_45(~bsxfun(@or, Ytrain==4, Ytrain==5), :) = [];
Y_45(Y_45==4) = 1;
Y_45(Y_45==5) = 0;

% Y_not_4 = Ytrain;
% Y_not_4(Ytrain==4) = [];   % (N-m)x1
% marker = find(Ytrain==4);
% X_not_4 = Xtrain;
% X_not_4(Ytrain==4, :) = [];

n_fold = 8;
partitions = make_xval_partition(size(X_45, 1), n_fold);
xval_train_set = X_45(partitions~=n_fold, :);
xval_train_label = Y_45(partitions~=n_fold);
xval_test_set = X_45(partitions==n_fold, :);
xval_test_label = Y_45(partitions==n_fold);



[~, info_4] = train_svm_audio(xval_train_set, xval_train_label, xval_test_set, xval_test_label);

% [~, info_not_4] = train_svm(X_not_4, Y_not_4);




% %% Train the other labels
% [~, info] = train_svm(X_not_4, Y_not_4);