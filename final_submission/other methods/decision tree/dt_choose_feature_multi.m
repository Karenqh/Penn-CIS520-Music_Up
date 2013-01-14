function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% YOUR CODE GOES HERE

H = multi_entropy(mean(Z));

% Compute conditional entropy for each feature.
ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

% Compute the IG of the best split with each feature. This is vectorized
% so that, for each feature, we compute the best split without a second for
% loop. Note that if we were guaranteed binary features, we could vectorize
% this entire loop with the same procedure.
t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', size(Z,1));
for i = colidx
    t.timeleft();

    % Check for constant values.
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    % Compute up to 10 possible splits of the feature.
    r = linspace(Xrange{i}(1), Xrange{i}(end), min(10, numel(Xrange{i})));
    split_f = bsxfun(@le, X(:,i), r(1:end-1));
    
    % Compute conditional entropy of all possible splits.
    px = mean(split_f);
    
%    cond_H=zeros(1,size(r(1:end-1),2));
    to_multi_1=zeros(size(Z,2),size(split_f,2));
    to_multi_2=zeros(size(Z,2),size(split_f,2));
    for j=1:size(Z,2)
        Y=Z(:,j);
        y_given_x = bsxfun(@and, Y, split_f);
        y_given_notx = bsxfun(@and, Y, ~split_f);
        to_multi_1(j,:)=sum(y_given_x)./sum(split_f);
        to_multi_2(j,:)=sum(y_given_notx)./sum(~split_f);
    end

    cond_H = px.*multi_entropy(to_multi_1')+ ...
            (1-px).*multi_entropy(to_multi_2');

    % Choose split with best IG, and record the value split on.
    [ig(i) best_split] = max(H-cond_H);
    split_vals(i) = r(best_split);
end

% Choose feature with best split.
[max_ig fidx] = max(ig);
val = split_vals(fidx);