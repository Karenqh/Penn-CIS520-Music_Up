function [loss scores test_err] = train_svm_audio_kmod(Xtrain, Ytrain, Xtest, Ytest)
% Scale the audio_vector (different units)
scale_mean = mean(Xtrain, 1);
Xtrain = bsxfun(@minus, Xtrain, scale_mean);
Xtest = bsxfun(@minus, Xtest, scale_mean);

% scale_sum = sum(Xtrain, 1);
scale_max = max(abs(Xtrain));
Xtrain = bsxfun(@rdivide, Xtrain, scale_max);
Xtest = bsxfun(@rdivide, Xtest, scale_max);
% Better use the scaling tool built in Libsvm


% % % What if we only use Timbre
% Xtrain = Xtrain(:, 7:end);
% Xtest = Xtest(:, 7:end);
Xtrain(:, 5) = [];
Xtest(:, 5) = [];

% % Cross validation to choose the best gamma/C  (grid search)
% bestcv = 0;
% bestc = 0.001;
% bestg = 0.0001;
% 
% matlabpool open;
% for log10c = 4:2:8,
%   parfor log10g = 3:7,
%     cmd = ['-b 1 -v 8 -c ', num2str(10^log10c), ' -g ', num2str(10^log10g)];
%     cv = svmtrain(Ytrain, Xtrain, cmd);
% %     if (cv >= bestcv),
% %       bestcv = cv; bestc = 10^log10c; bestg = 10^log10g;
% %     end
% %     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log10c, log10g, cv, bestc, bestg, bestcv);
%     fprintf('%g %g %g \n', log10c, log10g, cv);
%   end
% end
% matlabpool close;


% % The best parameters are: C = 10^4, gamma = 10^3;
% cmd = ['-b 1 -c ', num2str(10^4), ' -g ', num2str(10^3)];
% bestcv = svmtrain(Ytrain, Xtrain, cmd);



% % Use knn on features 3,4,5
% matlabpool open;
% parfor dummy = -4:3
%     log10c = 2*dummy;
%     cmd = ['-b 1 -t 0 -v 8 -c ', num2str(10^log10c)];
%     cv = svmtrain(Ytrain, Xtrain, cmd);
%     fprintf('%g accuracy=%g)\n', log10c, cv)
% end
% matlabpool close;

% Best C for linear kernel: 1???
% cmd = ['-b 1 -t 0 -c ', num2str(1)];
% bestcv = svmtrain(Ytrain, Xtrain, cmd);
% 
% model = bestcv;
% 
% % Use the bestcv result to make prediction on mock quiz dataset
% [yhat acc vals] = svmpredict(Ytest, Xtest, model, '-b 1');

kernel = @(x,x2) kernel_kmod(x, x2,120,4.5);
[xval_err info] = kernel_libsvm(Xtrain, Ytrain, Xtest, Ytest, kernel);

test_err = mean(info.yhat~=Ytest);

scores = zeros(size(Xtest,1), 10);
label_seq = (info.model.Label)';
for i = 1:size(Xtest,1)
    scores(i, label_seq) = info.vals(i, :);
end
ranks = get_ranks(scores);
loss = rank_loss(ranks, Ytest)


% % Further consider class 4 or 5
% X_45 = Xtest;
% X_45(~bsxfun(@or, yhat==4, yhat==5), :) = [];
% % Feed X_45 into decision tree clf
% % yhat45 = 
% ranks(bsxfun(@and, yhat==5, yhat45==1), 1) = 4;
% ranks(bsxfun(@and, yhat==4, yhat45==0),1) = 5;





