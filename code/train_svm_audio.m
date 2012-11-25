function [test_err model] = train_svm_audio(Xtrain, Ytrain, Xtest, Ytest)
% Scale the audio_vector (different units)
Xtrain = bsxfun(@rdivide, Xtrain, sum(Xtrain, 1));

% Cross validation to choose the best gamma/C  (grid search)
bestcv = 0;
bestc = 0.001;
bestg = 0.0001;
for log10c = -3:4,
  for log10g = -4:3,
    cmd = ['-b 1 -v 8 -c ', num2str(10^log10c), ' -g ', num2str(10^log10g)];
    cv = svmtrain(Ytrain, Xtrain, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 10^log10c; bestg = 10^log10g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log10c, log10g, cv, bestc, bestg, bestcv);
  end
end

model = bestcv;

% Use the bestcv result to make prediction on mock quiz dataset
[yhat acc vals] = svmpredict(Ytest, Xtest, model, '-b 1');

test_err = mean(yhat~=Ytest);

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



