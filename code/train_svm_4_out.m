function train_svm_4_out(Xtrain, Ytrain)


% Pick out genre 4
Y_4 = ones(length(Ytrain),1);
Y_4(Ytrain~=4) = 0;  %Nx1
Y_not_4 = Ytrain;
Y_not_4(Ytrain==4) = [];   % (N-m)x1
marker = find(Ytrain==4);
X_not_4 = Xtrain;
X_not_4(Ytrain==4, :) = [];

[test_err, info_4] = train_svm(Xtrain, Y_4);

TP = info_4.yhat
FP



% %% Train the other labels
% [~, info] = train_svm(X_not_4, Y_not_4);