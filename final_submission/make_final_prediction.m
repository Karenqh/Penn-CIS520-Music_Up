function ranks = make_final_prediction(model, example)
% Uses your trained model to make a final prediction for a SINGLE example.
%
% Usage:
%
%   RANKS = MAKE_FINAL_PREDICTION(MODEL, EXAMPLE);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes your trained model (output from INIT_MODEL) and a SINGLE 
% example, and returns a ranking ROW VECTOR as explained in the project
% overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Your model should be loaded from disk in INIT_MODEL. DO NOT DO
% ANY TRAINING HERE.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

% We only take in one example at a time.
target_lyrics = make_lyrics_sparse(example, model.vocab);
target_lyrics(:, model.cut_under_3) = [];

% scale
target_lyrics = target_lyrics./sum(target_lyrics);
% target_lyrics = bsxfun(@rdivide, target_lyrics, sum(target_lyrics, 2));

target_audio = make_audio(example);


% Prediction using SVM with lyrics
Ytest = 1;  % this is just a meaningless dummpy thing
% model.Xt_lyrics = bsxfun(@rdivide, model.Xt_lyrics, sum(model.Xt_lyrics, 2));

Ktest = kernel_intersection(model.Xt_lyrics, target_lyrics);
[~, ~, vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model.svm_model, '-b 1');
score_lyrics = zeros(1, 10);
label_seq = (model.svm_model.Label)';
score_lyrics(1, label_seq) = vals(1, :);


% Prediction using Neural Network with audio 
score_audio =  nn_audio(target_audio, model.mean_d, model.max_d, model.Theta1, model.Theta2);


% Combine the two results
weight_for_lyrics = 0.554; %!!!!!!!!!!!!!!!!!!!
weight_for_audio = 0.446;  %!!!!!!!!!!!!!!!!!!!
scores = score_lyrics.*weight_for_lyrics + score_audio.*weight_for_audio;
% 
% scores = score_lyrics;

% Convert into ranks
ranks = get_ranks(scores);

