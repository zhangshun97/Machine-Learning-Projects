function Y = manifold_pca(X, d)
% Input:
%     X: a matrix of size D x N, where D is the dimensinality of data and N
%     is the number of data points
% Output:
%     Y: a matrix of size d x N, where d is the main components derived by
%     pca

N = size(X,2);
Xctr = X - repmat(mean(X, 2), 1, N);
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

mainV = Vctr(:,end-d+1:end);
Y = mainV' * X;
