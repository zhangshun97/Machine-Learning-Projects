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

% 1.2.1
% derive the k for level 0.99 as always used in graphic applications
level = 0.99;

% centralized version
ssum = sum(lambda_ctr);
sssum = 0;
pvr = zeros(length(lambda_ctr),1);
for k = 1:length(lambda_ctr)
    sssum = sssum + lambda_ctr(end-k+1);
    pvr(k) = sssum/ssum;
end

clf;
% display the percentage w.r.t different k
plot(1:length(lambda_ctr),pvr);
hold on
line([203,203],[0.99,0],'linestyle',':','color','r');
line([0,203],[0.99,0.99],'linestyle',':','color','r');
scatter(203,0.99);

