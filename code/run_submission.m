clear;
load music_dataset.mat 
% load Xtest.mat 
% load Xtrain.mat 
% load Ytest.mat
% load Ytrain.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

% Xt_audio = make_audio(train);
% Xq_audio = make_audio(quiz);

%% Train SVM clfs
% Split the data into train(80%)/test(20%) set
% split_idx = make_xval_partition(size(train, 2), 5);
% Xtrain = Xt_lyrics(split_idx~=5, :); 
% Xtest = Xt_lyrics(split_idx==5, :);
% 
% Ytrain = Yt(split_idx~=5);
% Ytest = Yt(split_idx==5);
% 
% 
% save('Xtrain.mat', 'Xtrain');
% save('Ytrain.mat', 'Ytrain');
% save('Xtest.mat', 'Xtest');
% save('Ytest.mat', 'Ytest');

% [test_err info] = train_genre(Xtrain, Ytrain, Xtest, Ytest);


%% Run algorithm
ranks = predict_genre(Xtrain, Xtest, Ytrain);
%                       Xt_audio, Xq_audio, ...
%                       Yt);

% Got the rank loss
result = rank_loss(ranks, Ytest)

%% Generate submissions
ranks = predict_genre(Xt_lyrics, Xq_lyrics, Yt);
%                       Xt_audio, Xq_audio, ...
%                       Yt);

% Save results to a text file for submission
save('-ascii', 'submit.txt', 'ranks');

