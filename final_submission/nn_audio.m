function [pred]=nn_audio(X_test, mean_d, max_d, Theta1, Theta2)

%     load Theta1;
%     load Theta2;
%     load mean_d;
%     load max_d;
    
    X_test=(X_test-repmat(mean_d,size(X_test,1),1))./repmat(max_d,size(X_test,1),1);
    pred = predict(Theta1, Theta2, X_test);
%    nn_ranks = get_ranks(pred);
end
