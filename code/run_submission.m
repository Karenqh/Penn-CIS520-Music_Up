clear;
% pack;
load music_dataset.mat 
% load Xtest.mat 
% load Xtrain.mat 
% load Ytest.mat
% load Ytrain.mat

% [Xt_lyrics] = make_lyrics_sparse(train, vocab);
% [Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);

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

% split_idx = make_xval_partition(size(train, 2), 5);
% Xtrain_audio = Xt_audio(split_idx~=5, :); 
% Xtest_audio = Xt_audio(split_idx==5, :);
% 
% Ytrain_audio = Yt(split_idx~=5);
% Ytest_audio = Yt(split_idx==5);
% 
% 
% save('Xtrain_audio.mat', 'Xtrain_audio');
% save('Ytrain_audio.mat', 'Ytrain_audio');
% save('Xtest_audio.mat', 'Xtest_audio');
% save('Ytest_audio.mat', 'Ytest_audio');


% [test_err info] = train_svm(Xtrain, Ytrain, Xtest, Ytest);

%% "Train" just the intersection kernal
best_loop_size = just_kernel_train(Xtrain, Ytrain);
% It turns out that the best loop size is exactly 10...

%% Run algorithm
% ranks = predict_genre(Xtrain, Xtest, Ytrain);
ranks = predict_genre(Xtrain_audio, Xtest_audio, Ytrain_audio);

% Got the rank loss
result = rank_loss(ranks, Ytest_audio)

% %% Generate submissions
% ranks = predict_genre(Xt_lyrics, Xq_lyrics, Yt);
% %                       Xt_audio, Xq_audio, ...
% %                       Yt);
% 
% % Save results to a text file for submission
% save('-ascii', 'submit.txt', 'ranks');

