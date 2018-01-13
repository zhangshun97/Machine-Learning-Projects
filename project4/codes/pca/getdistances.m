function Y = getdistances(X)
% Input:
%    X: a matrix of size D x N, where D is the dimensionality and N
% is the number of data points.
%
% Output:
%    Y: a matrix of size N x N, of which the elements are the distances
%    from point to point in D-dimension space.

N = size(X,2);
Y = zeros(N,N);
for i = 1:N
    for j = 1:N
        Y(i,j) = norm(X(:,i)-X(:,j));
    end
end