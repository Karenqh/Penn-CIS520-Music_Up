function [loss] = rank_loss(ranks, Ytest)
loss = 0;
for i = 1:size(ranks,1)
    [~, idx] = find(ranks(i, :) == Ytest(i));
    loss = loss + 1 - 1/idx;
end
loss = loss/size(ranks,1);
