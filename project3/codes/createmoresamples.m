load digits.mat

[n,~] = size(X);
m = 100000; % number of created training samples
Xnew = zeros(m,256);
ynew = zeros(m,1);

for ii = 1:m
    i = ceil(rand*n);
    Xi = X(i,:);
    
    % detach bias and reshape
    Xt = reshape(Xi,16,16);

    % X will be translated slightly by random steps from 0 to 2
    % also random with up/down/left/right
    a = randi(5,1)-3;
    b = randi(5,1)-3;
    Xt = translate(Xt,a,b);

    % X will be rotated slightly by random degrees from 0 to 5
    % also random with clockwise or anti-clockwise
    theta = rand(1)*10 - 5;
    Xt = imrotate(Xt, theta);
    ss = size(Xt);
    Xt = imresize(Xt,16/ss(1));
    Xt = reshape(Xt,1,256);
    
    Xnew(ii,:) = Xt;
    ynew(ii) = y(i);
end
X = [X;Xnew];
y = [y;ynew];

save digits_more X Xtest Xvalid y ytest yvalid