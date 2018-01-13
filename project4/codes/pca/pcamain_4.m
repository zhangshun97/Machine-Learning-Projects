load freyface.mat
X = double(X);
N = size(X,2);
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun));
Vun = Vun(:, order);
Xctr = X - repmat(mean(X, 2), 1, N);
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

% 1.2.4
rx = 2000 * randn(2,1);
Y = Vctr(:,end-1:end) * rx;
Y = Y + repmat(mean(X, 2), 1, 1);
showfreyface(Y);
