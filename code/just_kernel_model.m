function ranks = just_kernel_model(xtrain, xtest, ytrain, loop_size)
scores = zeros(size(xtest, 1), 10);
D = kernel_intersection(xtrain, xtest);
% D = (xtest*xtrain').^2;
% D = kernel_gaussian(xtrain, xtest, 0.3);
Dtemp = D;
for j = 1:loop_size  
    [~, idx] = max(Dtemp, [], 2);
    ynn = idx(:, 1);
    yhat = ytrain(ynn);
    setmin = bsxfun(@eq, ytrain',yhat);
    Dtemp(setmin) = -999;

    for i=1:size(xtest, 1)
        scores(i, yhat(i)) = -j;
    end
end
ranks = get_ranks(scores);

