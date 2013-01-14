clear;
% pack;
load music_dataset.mat 
% load cut_idx.mat
% load cutter_idx.mat
% load cut_more.mat
% load cut_under_4
load cut_under_3.mat

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

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);


% Clean the vocabulary by eliminating words that accur only few times
% threshold tried: < 1, 2, 4, 5
% Xt_lyrics(:, cut_under_3) = [];
% Xq_lyrics(:, cut_under_3) = [];


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end


%% Train SVM clfs
% Split the data into train(80%)/test(20%) set
iter = 10;
loss = zeros(iter,1);
for xval = 1:iter
    xval
    
    split_idx = make_xval_partition(size(train, 2), 5);
%     Xtrain = Xt_lyrics(split_idx~=5, :); 
%     Xtest = Xt_lyrics(split_idx==5, :);
% 
%     Ytrain = Yt(split_idx~=5);
%     Ytest = Yt(split_idx==5);

    % split_idx = make_xval_partition(size(train, 2), 5);
    Xtrain_audio = Xt_audio(split_idx~=5, :); 
    Xtest_audio = Xt_audio(split_idx==5, :);

    Ytrain_audio = Yt(split_idx~=5);
    Ytest_audio = Yt(split_idx==5);

    % Train SVM only on lyrics
    % [info_4] = train_svm_4_out(Xtrain_audio, Ytrain_audio);
%     [loss_lyrics score_lyrics lyric_err] = test_svm(Xtrain, Ytrain, Xtest, Ytest);
%     loss(xval) = loss_lyrics;

    % Train SVM only on audio
    [loss_audio, ~, test_err] = train_svm_audio_kmod(Xtrain_audio, Ytrain_audio, Xtest_audio, Ytest_audio);
    loss(xval) = loss_audio;
end
loss_mean = mean(loss)

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



