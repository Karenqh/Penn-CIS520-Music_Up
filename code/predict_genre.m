function ranks = predict_genre(Xt_audio, Xq_audio,Ytrain)
%                                 Xt_lyrics, Xq_lyrics, 
%                                Yt)
% Returns the predicted rankings, given lyric and audio features.
%
% Usage:
%
%   RANKS = PREDICT_GENRE(XT_LYRICS, YT_LYRICS, XQ_LYRICS, ...
%                         XT_AUDIO, YT_AUDIO, XQ_AUDIO);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of lyric and audio features and produces a
% ranking matrix as explained in the project overview. 
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
% N = size(Xq_lyrics, 1);
% scores = zeros(N, 10);

% Scale the feature vector
% Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
% Xq = bsxfun(@rdivide, Xq_lyrics, sum(Xq_lyrics, 2));

Xt = bsxfun(@rdivide, Xt_audio, sum(Xt_audio, 1));
Xq = bsxfun(@rdivide, Xq_audio, sum(Xq_audio, 1));

ranks = just_kernel_model(Xt, Xq, Ytrain, 10);
% ranks = get_ranks(scores);


