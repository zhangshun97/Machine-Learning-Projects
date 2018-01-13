function [y] = MLPclassificationPredict_conv(w,X,nHidden,nLabels,kernel)

Xini = X;
len = size(X);
len = len(1);
X = zeros(len,145);
kernel = reshape(kernel,5,5);
kernel = imrotate(kernel,180);
for ii=1:length(Xini)
    Xi = Xini(ii,:);
    Xi = Xi(2:end);
    Xi = reshape(Xi,16,16);
    % convolution
    Xconv = conv2(Xi,kernel,'valid');
    Xi = reshape(Xconv,1,144);
    X0 = [1 Xi];
    X(ii,:) = X0;
end

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

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
