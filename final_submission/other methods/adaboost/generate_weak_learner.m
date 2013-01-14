function [ Yw ] = generate_weak_learner( Xtrain,Ytrain)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Yw = ceil(10*rand(size(Xtrain)));
Yw = ceil(10*rand(size(Xtrain,1),2000));
% for i = 1:size(Xtrain,2)
for i = 1:2000
    i
    kern_poly = @(x,x2) kernel_poly(x,x2,1);
    idx = randperm(size(Xtrain,1));
    train = Xtrain(idx(1:1000),:);
    train_label = Ytrain(idx(1:1000));
    test = Xtrain(idx(1001:end),:);
    test_label = Ytrain(idx(1001:end));
    [x_err info] = kernel_libsvm(train, train_label,...
                                    test, test_label, kern_poly);
     Yw(idx(1001:end),i) = info.yhat;
    
                                

end
% for i=1:10
%     class(i) = struct('value',full(Xtrain(Ytrain == i,:)));
% 
% end
% 
% for i=1:10
%     [count index] = max(class(i).value,[],2);
%     class(i).count = count;
%     class(i).index = index;
% end
% 
% for i = 1:10
%     for j = 1:15
%         [temp_count temp_idx ] = max(class(i).count);
%         class(i).count(class(i).index == class(i).index(temp_idx)) = -999;
% %         class(i).count(temp_idx) = -999;
%         feature(i,j) = class(i).index(temp_idx);
%     end
% 
% end
% 
% Yw = ceil(10*rand(size(Xtrain)));
% 
% for i = 5000:6000
%     Yw(i,:) = 5;
% end
% 
% for i = 1:size(Xtrain,1)
% % for i = 1:2
%    for j = 1:10
%        for k = 1:3
%            if(Xtrain(i,selectedfeatures(j,k)) ~= 0)
%                Yw(i,selectedfeatures(j,k)) = j;
%            end
% %            Yw(i,Xtrain(i,:) == selectedfeatures(j,k)) = j;
%        end
%    end
% end
% Xnew = zeros(30,length(Ytrain));
% for i = 1:length(Ytrain)
% %     for j = 1:10
% %         if (find(Xtrain == selectedfeatures(j,:)))
% %             weight(j) = 1/find(Xtrain == selectedfeatures(j,:),1,'first');
% %         end
% %     end
% %     
%     for j = 1:30
%         if(find(Xtrain == selectedfeatures(j)))
%             Xnew(Xtrain == selectedfeatures(j))= 1;
%         end
%         
%     end
%     
%         
%    

end