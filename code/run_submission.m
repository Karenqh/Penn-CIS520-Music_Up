clear;
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end
Ytest = ones(size(Xq_lyrics,1),1);
Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);
% partition = make_xval_partition(size(Xt_lyrics,1),5);
% Xtrain = Xt_lyrics(partition<=4,:);
% Ytrain = Yt(partition<=4);
% 
% Xtest = Xt_lyrics(partition ==5,:);
% Ytest = Yt(partition == 5);
% k_intersection = @(x,x2) kernel_intersection(x,x2);
% [test_err info] = kernel_libsvm(Xtrain,Ytrain,Xtest,Ytest,k_intersection);
% k_quad = @(x,x2) kernel_poly(x,x2,2);
% [test_err info] = kernel_libsvm(Xt_lyrics,Yt,Xq_lyrics,Ytest,k_quad);
%% Run algorithm
ranks = predict_genre(Xt_lyrics, Xq_lyrics, ...
                      Xt_audio, Xq_audio, ...
                      Yt);

% Save results to a text file for submission
save('-ascii', 'submit.txt', 'ranks');

