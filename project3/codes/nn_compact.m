load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [100];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);
% decay
lambda = 0;

tic;
% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
stepSize2 = 1e-5;
funObj = @(w,Xi,yi)MLPclassificationLoss_vectorized(w,Xi,yi,nHidden,nLabels,lambda);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    % stpchastic gradient descent
    i = ceil(rand*n);
    [~,g,yhat] = funObj(w,X(i,:),yExpanded(i,:));
    w = w - stepSize*g;
    [~,g1,~] = funObj(w,compactsample(X(i,:)),yhat);
    w = w - stepSize2*g1;
end
toc;

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);