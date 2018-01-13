function [f,g] = MLPclassificationLoss_moresamples(w,X,y,nHidden,nLabels)

% only works for stochastic gradient descent
% that is to say, X's size is (1,257)

% detach bias and reshape
X = reshape(X(2:end),16,16);

% X will be translated slightly by random steps from 0 to 2
% also random with up/down/left/right
a = randi(5,1)-3;
b = randi(5,1)-3;
X = translate(X,a,b);

% X will be rotated slightly by random degrees from 0 to 5
% also random with clockwise or anti-clockwise
theta = rand(1)*10 - 5;
X = imrotate(X, theta);
ss = size(X);
X = imresize(X,16/ss(1));
X = reshape(X,1,256);
% add a bias
X = [1 X];

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    % forward process
    % input layer
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    % hidden layers
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    % output value
    yhat = fp{end}*outputWeights;
    
    % loss term
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        gOutput = gOutput + fp{end}' * err;

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                backprop(c,:) = err(c)*(sech(ip{end}).^2.*outputWeights(:,c)');
                gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:);
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
        else
            % Input Weights
            gInput = X(i,:)' * (sech(ip{end}).^2.*(err * outputWeights'));
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
