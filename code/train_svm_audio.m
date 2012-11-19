function train_svm_audio(Xtrain, Ytrain)
% Scale the audio_vector (different units)
Xtrain = bsxfun(@rdivide, Xtrain, sum(Xtrain, 1));

% Cross validation to choose the best sigma
sigma = [1 5 10 15 20];  % what will be a proper sigma?
for i = 1:length(sigma)
    % train svm to get error for a certain sigma
    % maybe I should use the built-in RBF kernel and the built in cv
    for cv = 1:10
        % An iner
    end
end
