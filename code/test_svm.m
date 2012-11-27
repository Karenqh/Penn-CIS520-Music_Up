function [loss_new scores test_err] = test_svm(Xtrain, Ytrain, Xtest, Ytest, Xaudio, tree)
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
loss_old = rank_loss(ranks, Ytest)

% % Further consider class 4 or 5
% X_45 = Xaudio;
% X_45(~bsxfun(@or, yhat==4, yhat==5), :) = [];
% % Feed X_45 into decision tree clf
% yhat45 = dt_audio_45(X_45, tree);

% ydumb = 11.*ones(length(yhat),1);
% ydumb(cat(1, find(yhat==4), find(yhat==5))) = yhat45;
% 
% count = 0;
% for i = 1:length(Ytest)
%     if yhat(i)==4 || yhat(i)==5
%         count = count + 1;
%         if yhat(i)==4 && yhat45(count) == 0
%             ranks(i,1) = 5;
%             temp = ranks(i,:);
%             temp(find(temp==5, 1, 'last')) =4;
%             ranks(i,:) = temp;
% %             temp = cat(2, 5, ranks(i,:));
% %             temp(find(temp==5, 1, 'last')) = [];
% %             ranks(i, :)= temp;
%         elseif yhat(i)==5 && yhat45(count)==1
%             ranks(i,1) = 4;
%             temp = ranks(i,:);
%             temp(find(temp==4, 1, 'last')) =5;
%             ranks(i,:) = temp;
% %             temp = cat(2, 4, ranks(i,:));
% %             temp(find(temp==4, 1, 'last')) = [];
% %             ranks(i, :)= temp;
%         end
%     end
% end

loss_new = rank_loss(ranks, Ytest)

% Optionally we can look at more information from training/testing.
info.vals = vals;
info.yhat = yhat;
info.model = model;

