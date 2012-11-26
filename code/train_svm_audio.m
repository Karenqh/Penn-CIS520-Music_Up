function [loss scores model] = train_svm_audio(Xtrain, Ytrain, Xtest, Ytest)
% Scale the audio_vector (different units)
scale_mean = mean(Xtrain, 1);
Xtrain = bsxfun(@minus, Xtrain, scale_mean);
Xtest = bsxfun(@minus, Xtest, scale_mean);

% scale_sum = sum(Xtrain, 1);
scale_max = max(abs(Xtrain));
Xtrain = bsxfun(@rdivide, Xtrain, scale_max);
Xtest = bsxfun(@rdivide, Xtest, scale_max);
% Better use the scaling tool built in Libsvm

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

% What if we only use Timbre
Xtrain = Xtrain(:, 7:end);
Xtest = Xtest(:, 7:end);

% The best parameters are: C = 10^4, gamma = 10^3;
cmd = ['-b 1 -c ', num2str(10^4), ' -g ', num2str(10^3)];
bestcv = svmtrain(Ytrain, Xtrain, cmd);

model = bestcv;

% Use the bestcv result to make prediction on mock quiz dataset
[yhat acc vals] = svmpredict(Ytest, Xtest, model, '-b 1');

test_err = mean(yhat~=Ytest)

scores = zeros(size(Xtest,1), 10);
label_seq = (model.Label)';
for i = 1:size(Xtest,1)
    scores(i, label_seq) = vals(i, :);
end
ranks = get_ranks(scores);
loss = rank_loss(ranks, Ytest)


% gamma = [1 5 10 15 20];  % what will be a proper sigma?
% for i = 1:length(sigma)
%     % train svm to get error for a certain sigma
%     % maybe I should use the built-in RBF kernel and the built in cv
%     for cv = 1:10
%         % An iner
%     end
% end



