load digits.mat
[n,~] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = 144 + 1;

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
w0 = w;
% penalty
lambda = 0.05;
lambda_conv = 0.005;
% a common choice of the filter is (5,5)
kernel = rand(25,1);

tic;
% Train with stochastic gradient
maxIter = 100000;
stepSize = 5e-4;
stepSize_conv = 1e-5;
funObj_conv = @(w,i)MLPclassificationLoss_convolution_decay(w,X(i,:),yExpanded(i,:),nHidden,nLabels,kernel,lambda,lambda_conv);
for iter = 1:maxIter
    % Validation error
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict_conv(w,Xvalid,nHidden,nLabels,kernel);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    % Loss function
    i = ceil(rand*n);
    [f,g,gConv] = funObj_conv(w,i);
    kernel = kernel - stepSize_conv*gConv;
    
    if iter < 2
        w1 = w0 - stepSize*g;
        w = w1;
        wt_1 = w0;
    else
        wt_new = w - stepSize*g + 0.9*(w - wt_1);
        
        wt_1 = w;
        w = wt_new;
    end
end

toc;
% Evaluate test error
yhat = MLPclassificationPredict_conv(w,Xtest,nHidden,nLabels,kernel);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
