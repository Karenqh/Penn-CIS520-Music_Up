function [yhat] = dt_audio_45(X_test,tree)

label_pre=zeros(size(X_test,1),1);

for j=1:size(X_test,1)
    label_pre(j)=dt_value(tree,X_test(j,:));
end

label=(label_pre>=0.5);
yhat = zeros(length(X_test), 1);
yhat(label) = 1;
end
