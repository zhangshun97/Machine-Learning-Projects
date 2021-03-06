function [model] = modified_svm(X,y,lambda,maxIter)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Matlab indexes by columns,
%  so if we are accessing rows it will be faster to use  the traspose
Xt = X';

% Initial values of regression parameters
w = zeros(d+1,1);
wmean = w;

% Apply stochastic gradient method
for t = 1:maxIter
    if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)
        if t < maxIter/2
            wmean = w;
        end
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*wmean))) + (lambda/2)*(wmean'*wmean);
        semilogy([0:t/n],objValues);
        pause(.1);
    end
    
    sgs = zeros(d+1, 1);
    miniter = 10;
    for tt = 1:miniter
        % Pick a random training example
        i = ceil(rand*n);

        % Compute sub-gradient
        [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i);
        
        sgs = sgs + sg/miniter;
    end
    sg = sgs;
    
    % Set step size
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient step
    w = w - alpha*(sg + lambda*w);
    if t >= maxIter/2
        tt = t - maxIter/2;
        wmean = wmean*tt/(tt+1) + w/(tt+1);
    end
end

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function [f,sg] = hingeLossSubGrad(w,Xt,y,lambda,i)

[d,n] = size(Xt);

% Function value
wtx = w'*Xt(:,i);
loss = max(0,1-y(i)*wtx);
f = loss;

% Subgradient
if loss > 0
    sg = -y(i)*Xt(:,i);
else
    sg = sparse(d,1);
end
end

