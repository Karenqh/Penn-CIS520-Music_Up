function ranks = predict_genre(Xt_lyrics, Xq_lyrics, Ytrain)
%                                Xt_audio, Xq_audio, 
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
N = size(Xq_lyrics, 1);
scores = zeros(N, 10);

% Scale the feature vector
Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
Xq = bsxfun(@rdivide, Xq_lyrics, sum(Xq_lyrics, 2));

% D = Xq*Xt'; % Poly kernel
D = kernel_intersection(Xt, Xq);  % Intersection kernel using hist
Dtemp = D;
for j = 1:10
    [~, idx] = max(Dtemp, [], 2);
    ynn = idx(:, 1);
    yhat = Ytrain(ynn);
    setmin = bsxfun(@eq,Ytrain',yhat);
    Dtemp(setmin) = -999;
   
    for i=1:N
        scores(i, yhat(i)) = 11 -j;
    end
end

ranks = get_ranks(scores);
end

