function [f,g,gConv] = MLPclassificationLoss_convolution(w,X,y,nHidden,nLabels,kernel)

% !!! only with stochastic gradient descent
X0 = X(2:end);
X = reshape(X0,16,16);
kernel = reshape(kernel,5,5);
% convolution
Xconv = conv2(X,imrotate(kernel,180),'valid');
X = reshape(Xconv,1,144);
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
    fpt = tanh(X(i,:));
    ip{1} = fpt*inputWeights;
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
        
        % use for single hidden layer for convolution layer
        backprop = (sech(ip{end}).^2.*(err * outputWeights'));

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
            gInput = gInput + fpt'*backprop;
        else
            % Input Weights
            gInput = gInput + fpt' * (sech(ip{end}).^2.*(err * outputWeights'));
        end
        % backprop size = (1,144)
        backprop = (backprop*inputWeights').*sech(X(i,:)).^2;
        % detach bias
        backprop = reshape(backprop(2:end),12,12);
        % backward convolution
        gConv = conv2(reshape(X0,16,16),imrotate(backprop,180),'valid');
        gConv = reshape(gConv,25,1);
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
