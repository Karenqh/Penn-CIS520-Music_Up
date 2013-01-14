function [model_linear model_sig] = svm_audio(Xtrain, Ytrain, Xtest, Ytest)
% Scale the audio_vector (different units)
scale_mean = mean(Xtrain, 1);
Xtrain = bsxfun(@minus, Xtrain, scale_mean);
Xtest = bsxfun(@minus, Xtest, scale_mean);

% % scale_sum = sum(Xtrain, 1);
% scale_max = max(abs(Xtrain));
% Xtrain = bsxfun(@rdivide, Xtrain, scale_max);
% Xtest = bsxfun(@rdivide, Xtest, scale_max);

scale_min = min(Xtrain);
scale_norm = 0.5*(max(Xtrain) - min(Xtrain));
Xtrain = bsxfun(@minus, Xtrain, scale_min);
Xtrain = bsxfun(@rdivide, Xtrain, scale_norm);
Xtrain = bsxfun(@minus, Xtrain, 1);

Xtest = bsxfun(@minus, Xtest, scale_min);
Xtest = bsxfun(@rdivide, Xtest, scale_norm);
Xtest = bsxfun(@minus, Xtest, 1);


% % What if we only use Timbre
Xtrain(:, 1:6) = [];
Xtest(:, 1:6) = [];

Xtrain(:, 1:2:23) = [];
Xtest(:, 1:2:23) = [];

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

% % Best C for linear kernel: 1???
% cmd_linear = ['-b 1 -t 0 -c ', num2str(10)];
% bestlinear = svmtrain(Ytrain, Xtrain, cmd_linear);


% % matlabpool open;
% for log10c = 3:4
%     % log10g = -2 is the best
%     % g = ~ -0.01
%         best dummy = 1
%   parfor dummy = 0:3
%       g = 5*10^(-3+0.2*dummy);
%     cmd = ['-b 1 -t 3 -v 8 -c ', num2str(10^log10c), ' -g ', num2str(g)];
%     cv = svmtrain(Ytrain, Xtrain, cmd);
% %     if (cv >= bestcv),
% %       bestcv = cv; bestc = 10^log10c; bestg = 10^log10g;
% %     end
% %     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log10c, log10g, cv, bestc, bestg, bestcv);
%     fprintf('%g %g %g \n', 10^log10c, dummy, cv);
%   end
% end
% matlabpool close;

% Best C for linear kernel: 1???
g = 5*10^(-3+0.2);
cmd_sig = ['-b 1 -t 3 -c ', num2str(100), ' -g ', num2str(g)];
bestsig = svmtrain(Ytrain, Xtrain, cmd_sig);



% model_linear = bestlinear;
model_sig = bestsig;

% Use the bestcv result to make prediction on mock quiz dataset
% [~, ~, vals_linear] = svmpredict(Ytest, Xtest, model_linear, '-b 1');
% test_err = 0;

% score_linear = zeros(size(Xtest,1), 10);
% label_seq = (model_linear.Label)';
% for i = 1:size(Xtest,1)
%     score_linear(i, label_seq) = vals_linear(i, :);
% end
% ranks = get_ranks(score_linear);
% loss_linear = rank_loss(ranks, Ytest)
% 

[~, ~, vals_sig] = svmpredict(Ytest, Xtest, model_sig, '-b 1');
score_sig = zeros(size(Xtest,1), 10);
label_seq = (model_sig.Label)';
for i = 1:size(Xtest,1)
    score_sig(i, label_seq) = vals_sig(i, :);
end
ranks = get_ranks(score_sig);
loss_sig = rank_loss(ranks, Ytest)

% % Combine
% weight_linear = loss_sig/(loss_linear + loss_sig);
% weight_sig = loss_linear/(loss_linear + loss_sig);
% 
% scores = score_linear.*weight_linear + score_sig.*weight_sig;
% ranks = get_ranks(scores);
% loss = rank_loss(ranks, Ytest)
% 
% 




