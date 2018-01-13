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

% 1.2.3
%V = Vctr;
V = Vun;
Y = V(:,end-1:end)' * X;
plot(Y(1,:), Y(2,:), '.');
explorefreymanifold(Y, X);