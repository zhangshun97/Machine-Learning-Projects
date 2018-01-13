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
nHidden = [200];

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

% mini-batch size
m = 10;

tic;
% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-4;
funObj = @(w,i)MLPclassificationLoss_decay(w,X(i,:),yExpanded(i,:),nHidden,nLabels,lambda);
for iter = 1:maxIter
    % Validation error
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        enow = sum(yhat~=yvalid)/t;
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,enow);
        % early stopping
        if iter < 2
            elast = enow;
            wlast = w;
        elseif enow > elast
            w = wlast;
            break;
        else
            wlast = w;
            elast = enow;
        end
    
    end
    
    % Loss function for mini-batch
    gs = zeros(nParams,m);
    for mi = 1:m
        i = ceil(rand*n);
        [f,g] = funObj(w,i);
        gs(:,mi) = g;
    end
    g = mean(gs,2);
    
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
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
