clear;
% pack;
load music_dataset.mat 
load tree.mat

% load Xtest.mat 
% load Xtrain.mat 
% load Ytest.mat
% load Ytrain.mat
% 
% 
% load Xtrain_audio.mat
% load Xtest_audio.mat
% load Ytrain_audio.mat
% load Ytest_audio.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

% % Slim the vocabulary
% vocab{sum(Xt_lyrics)==0} = [];
% save('new_vocab.mat', vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);

%% Train SVM clfs
% Split the data into train(80%)/test(20%) set
loss = 0;
for xval = 1:10
    split_idx = make_xval_partition(size(train, 2), 5);
    Xtrain = Xt_lyrics(split_idx~=5, :); 
    Xtest = Xt_lyrics(split_idx==5, :);

    Ytrain = Yt(split_idx~=5);
    Ytest = Yt(split_idx==5);

    % 
    % save('Xtrain.mat', 'Xtrain');
    % save('Ytrain.mat', 'Ytrain');
    % save('Xtest.mat', 'Xtest');
    % save('Ytest.mat', 'Ytest');

    % split_idx = make_xval_partition(size(train, 2), 5);
    Xtrain_audio = Xt_audio(split_idx~=5, :); 
    Xtest_audio = Xt_audio(split_idx==5, :);

    Ytrain_audio = Yt(split_idx~=5);
    Ytest_audio = Yt(split_idx==5);


    % save('Xtrain_audio.mat', 'Xtrain_audio');
    % save('Ytrain_audio.mat', 'Ytrain_audio');
    % save('Xtest_audio.mat', 'Xtest_audio');
    % save('Ytest_audio.mat', 'Ytest_audio');

    % Train SVM only on lyrics
    % [info_4] = train_svm_4_out(Xtrain_audio, Ytrain_audio);
    % [loss_lyrics score_lyrics lyric_err] = test_svm(Xtrain, Ytrain, Xtest, Ytest, Xtest_audio, tree);

    % Train SVM only on audio
    [loss_audio score_audio audio_err] = train_svm_audio(Xtrain_audio, Ytrain_audio, Xtest_audio, Ytest_audio);
    loss = loss + loss_audio;
end
loss_mean = loss/10

% %% "Train" just the intersection kernal
% best_loop_size = just_kernel_train(Xtrain, Ytrain);
% % It turns out that the best loop size is exactly 10...

%% How to combine lyrics and audio
% lyrics give a Nx10 probas
% audio give a Nx10 probas

% According to rank loss or accuraccy??
norm_for_lyrics = loss_audio/(loss_lyrics + loss_audio);
norm_for_audio = loss_lyrics/(loss_lyrics + loss_audio);

% norm_for_lyrics = audio_err/(lyric_err+audio_err);
% norm_for_audio = lyric_err/(lyric_err+audio_err);

scores = score_lyrics.*norm_for_lyrics + score_audio.*norm_for_audio;
% scores = bsxfun(@max, score_lyrics.*norm_for_lyrics, score_audio.*norm_for_audio);

ranks = get_ranks(scores);
loss = rank_loss(ranks, Ytest)


%% Run algorithm
Yq = ones(size(Xq_lyrics, 1), 1);
[loss_lyrics score_lyrics lyric_model] = test_svm(Xt_lyrics, Yt, Xq_lyrics, Yq);
[loss_audio score_audio audio_model] = train_svm_audio(Xt_audio, Yt, Xq_audio, Yq);

norm_for_lyrics = 0.558;
norm_for_audio = 0.442;
scores = score_lyrics.*norm_for_lyrics + score_audio.*norm_for_audio;
ranks = get_ranks(scores);


% ranks = predict_genre(Xtrain_audio, Xtest_audio, Ytrain_audio);

% Got the rank loss
% result = rank_loss(ranks, Ytest_audio)

% %% Generate submissions
% ranks = predict_genre(Xt_lyrics, Xq_lyrics, Yt);
% %                       Xt_audio, Xq_audio, ...
% %                       Yt);
% 
% Save results to a text file for submission
save('-ascii', 'submit.txt', 'ranks');

