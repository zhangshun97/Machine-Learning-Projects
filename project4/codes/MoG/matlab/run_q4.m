load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];

rng(10)
for i = 1 : 4
    K = numComponent(i);
% Train a MoG model with K components for digit 2
%-------------------- Add your code here --------------------------------
    [p2,mu2,vary2,logProbX2] = mogEM_kmeans(train2,K,30,0.01,0);

% Train a MoG model with K components for digit 3
%-------------------- Add your code here --------------------------------
    [p3,mu3,vary3,logProbX3] = mogEM_kmeans(train3,K,30,0.01,0);

% Caculate the probability P(d=1|x) and P(d=2|x), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function
%-------------------- Add your code here --------------------------------
    s2 = sum(mogLogProb(p3,mu3,vary3,train2) > mogLogProb(p2,mu2,vary2,train2));
    s3 = sum(mogLogProb(p3,mu3,vary3,train3) < mogLogProb(p2,mu2,vary2,train3));
    errorTrain(i) = (s2+s3)/(length(train2)+length(train3));
    
    s2 = sum(mogLogProb(p3,mu3,vary3,valid2) > mogLogProb(p2,mu2,vary2,valid2));
    s3 = sum(mogLogProb(p3,mu3,vary3,valid3) < mogLogProb(p2,mu2,vary2,valid3));
    errorValidation(i) = (s2+s3)/(length(valid2)+length(valid3));
    
    s2 = sum(mogLogProb(p3,mu3,vary3,test2) > mogLogProb(p2,mu2,vary2,test2));
    s3 = sum(mogLogProb(p3,mu3,vary3,test3) < mogLogProb(p2,mu2,vary2,test3));
    errorTest(i) = (s2+s3)/(length(test2)+length(test3));
end

% Plot the error rate
%-------------------- Add your code here --------------------------------
plot(xs,errorTrain,'b',xs,errorValidation,'g',xs,errorTest,'r')
title('Errors of models with different number of clusters')
xlabel('number of clusters each model')
ylabel('average classification error rate')
legend('Train error','Validation error','Test error')
set(gca,'xtick',[2 5 15 25])
