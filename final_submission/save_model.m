% THIS IS NOT A FUNCTION.
%
% This is just a script that is called to train the model. It saves
% whatever we will need at test time.

% load('../data/music_dataset.mat');

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

clear;
load music_dataset.mat 
load cut_under_3.mat
load svm_model.mat
load Theta1;
load Theta2;
load mean_d;
load max_d;


% % Generate the sparse training set that we'll need for nearest neighbor
[Xt_lyrics] = make_lyrics_sparse(train, vocab);
Xt_lyrics(:, cut_under_3) = [];
Xt_lyrics = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));

Xt_audio = make_audio(train);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

%% SVM with lyrics
% Xtrain = Xt_lyrics;
% % scale
% Xtrain = bsxfun(@rdivide, Xtrain, sum(Xtrain, 2));
% K = kernel_intersection(Xtrain, Xtrain);
% 
% svm_model = svmtrain(Yt, [(1:size(K,1))' K], sprintf('-t 4 -c %g -b 1', 100));
% 
% Xtest = Xt_lyrics(2000:3000 ,:);
% Xtest = bsxfun(@rdivide, Xtest, sum(Xtest, 2));
% Ytest = Yt(2000:3000);
% Ktest = kernel_intersection(Xtrain, Xtest);
% [~, ~, vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], svm_model, '-b 1');
% 
% scores = zeros(size(Xtest,1), 10);
% label_seq = (svm_model.Label)';
% for i = 1:size(Xtest,1)
%     scores(i, label_seq) = vals(i, :);
% end
% ranks = get_ranks(scores);
% 
% loss = rank_loss(ranks, Ytest)


% save('svm_model.mat', 'svm_model');




%%
model.cut_under_3 = cut_under_3;
model.Xt_lyrics = Xt_lyrics;
model.svm_model = svm_model;
model.vocab = vocab;
model.Xt_audio = Xt_audio;
model.Theta1 = Theta1;
model.Theta2 = Theta2;
model.mean_d = mean_d;
model.max_d = max_d;





save('my_model.mat', 'model');




% What we need for final prediction
% svm_lyrics: svm_clf, Ktest(we need Xt_lyrics), cut_under_3
% nn
% combine: weight_for_lyrics, weight_for_audio



