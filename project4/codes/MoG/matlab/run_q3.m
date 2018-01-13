load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

K = 20;
minVary = 0.01;
iters = 20;

% Train with kmeans initialization
[p,mu,vary,logProbX] = mogEM_kmeans(x,K,iters,minVary,0);
disp(logProbX);

% Train with randConst=1 initialization
[p,mu,vary,logProbX] = mogEM_rconst(x,K,iters,minVary,0,1);
disp(logProbX);
