function [ boost ] = multiBoost_train(Y, Yw, T)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


errMat = ~bsxfun(@eq, Y, Yw);
D = ones(size(Y));
D(:) = 1./sum(D);

for t = 1:T
    t
    [err(t), h(t)] = min(sum(bsxfun(@times,D,errMat)));
    alpha(t) =log((1-err(t))/err(t)) + log(10-1);
    for i = 1:10
        acc(:,i) = sum(bsxfun(@times,alpha,Yw(:,h) == i),2);
    end
    [weight h_t] = max(acc,[],2);
    test_err(t) = mean(h_t ~= Y);
    temp = (Y == Yw(:,h(t)));
%     temp = temp*2;
%     temp = temp -1;

    z_t = D' * exp(-alpha(t) *temp);
    D = D.*exp(-alpha(t) * temp) / z_t;
end
boost.err = err;
boost.h = h;
boost.alpha = alpha;
boost.test_err = test_err;
boost.acc = acc;
end

