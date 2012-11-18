function loop_size = just_kernel_train(Xtrain, Ytrain)
% lrange = [10, 15, 20, 30];
lrange = [10, 11, 12];
loss = zeros(1, length(lrange));
for para = 1:length(lrange)
    % cross validation for 10 cycles
    for count = 1:10
        partitions = make_xval_partition(size(Xtrain, 1), 10);
        xval_train_set = Xtrain(partitions~=10, :);
        xval_train_label = Ytrain(partitions~=10);
        xval_test_set = Xtrain(partitions==10, :);
        xval_test_label = Ytrain(partitions==10);
        scores = zeros(size(xval_test_set, 10));
        
        % Scale the feature vector
        xval_train_set = bsxfun(@rdivide, xval_train_set, sum(xval_train_set, 2));
        xval_test_set = bsxfun(@rdivide, xval_test_set, sum(xval_test_set, 2));
        
        D = kernel_intersection(xval_train_set, xval_test_set);  % Intersection kernel using hist
        Dtemp = D;
        for j = 1:lrange(para)  % 10 is not enough.
            [~, idx] = max(Dtemp, [], 2);
            ynn = idx(:, 1);
            yhat = xval_train_label(ynn);
            setmin = bsxfun(@eq, xval_train_label',yhat);
            Dtemp(setmin) = -999;

            % what if the 2nd, 3rd time I got the same prediction??? This will
            % result in reducing the score of the correct label
            for i=1:size(xval_test_set, 1)
                scores(i, yhat(i)) = -j;
            end
        end
        ranks = get_ranks(scores);
        loss(para) = loss(para) + rank_loss(ranks, xval_test_label);
    end
    loss(para) = loss(para)/10
end
[~, idx] = min(loss);
loop_size = lrange(idx);

