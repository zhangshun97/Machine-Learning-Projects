load quantum.mat
[n,d] = size(X);

% Split into training and validation set
perm = randperm(n);
Xvalid = X(n/2+1:end,:);
yvalid = y(n/2+1:end);
X = X(1:n/2,:);
y = y(1:n/2);

n = n/2;
lambda = 1/n;
model = modified_svm(X,y,lambda,25*n);
hold on
y = zeros(1,25) + 1.0;
plot(y, '--r')
hold off
