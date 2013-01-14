%% Script/instructions on how to submit plots/answers for question 3.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data

% load Xtest.mat 
% load Xtrain.mat 
% load Ytest.mat
% load Ytrain.mat

load X_test_a.mat 
load X_train_a.mat 
load Y_test_a.mat
load Y_train_a.mat

% Running a training set for binary decision tree classifier
%[X Y] = get_digit_dataset(data, {'7','9'}, 'train');


%% 3.1


% [X_train Y_train] = get_digit_dataset(data, {'1','3','7'}, 'train');
% [X_test Y_test] = get_digit_dataset(data, {'1','3','7'}, 'test');




X_train=X_train_a;
Y_train=Y_train_a;
X_test=X_test_a;
Y_test=Y_test_a;

mean_d=mean(X_train);
max_d=max(X_train);


% mean_d=repmat(mean(X_train),size(X_train,1),1);
% sum_d=repmat(sum(X_train),size(X_train,1),1);

X_train=(X_train-repmat(mean_d,size(X_train,1),1))./repmat(max_d,size(X_train,1),1);
X_test=(X_test-repmat(mean_d,size(X_test,1),1))./repmat(max_d,size(X_test,1),1);

scores = zeros(size(Y_test,1), 10);

depth=9;
train_error=zeros(1,max(depth));
test_error=zeros(1,max(depth));

tree = dt_train_multi(X_train, Y_train, depth);
% for j=1:size(X_train,1)
%     temp_set=dt_value(tree,X_train(j,:));
%     if  find(temp_set==max(temp_set))~=Y_train(j)
%         train_error(i)=train_error(i)+1;
%     end
% end
for j=1:size(X_test,1)
    temp_set=dt_value(tree,X_test(j,:));
%     if find(temp_set==max(temp_set))~=Y_test(j)
%         test_error(i)=test_error(i)+1;
%     end
    scores(j,:)=temp_set;
    
end

% train_error=train_error/size(X_train,1);
% test_error=test_error/size(X_test,1);

% plot(depth,train_error,'o');
% hold on;
% plot(depth,test_error,'r*');
% legend('Train Error','Test Error');
% 
% title('Question 3.1')
% 
% xlabel('Depth');
% ylabel('Errors');
% 
% hold off;

M_error=zeros(10,10);

for j=1:size(X_test,1)
    temp_set=dt_value(tree,X_test(j,:));
    Y_temp=find(temp_set==max(temp_set));

     M_error(Y_test(j),Y_temp)=M_error(Y_test(j),Y_temp)+1;
    if Y_temp~=Y_test(j)
        Y_right=Y_test(j);
        Y_wrong=Y_temp;
        X_row=j;
    end
end

M_error=M_error/size(X_test,1);
plotnumeric(M_error);
genre_sum_dt=sum(M_error,2);
genre_rate_dt=zeros(10,1);
for i=1:10
    genre_rate_dt(i)=M_error(i,i)/genre_sum_dt(i);
end
% genre_sum_dt
% genre_rate_dt



ranks = get_ranks(scores);
result = rank_loss(ranks, Y_test)

