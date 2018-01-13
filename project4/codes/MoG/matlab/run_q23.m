load digits;
x = [train2, train3];
%-------------------- code for 2.3 --------------------------------

%colormap(gray);
%imagesc(reshape(train2(:,1),16,16));
clf;
n = 10;
randConst = 1;
logPs = zeros(n,1);
for i = 1:1
    [p,mu,vary,logProbX] = mogEM_rconst(x,20,30,0.01,0,randConst);
    [p,mu,vary,logProbXk] = mogEM_kmeans(x,20,30,0.01,0);
    %logPs(i) = logProbX(end);
end
%mlogP = mean(logPs);

plot(1:30,logProbX,'b',1:30,logProbXk,'r')
legend('randConst','k-means')
ylabel('log-prob')
xlabel('iteration')
title('training process')

%plot(1:n, logPs, 'r');
%line([1, 10],[mlogP, mlogP]);
%axis([1 10 -15000 -2000]);
%set(gca, 'ytick', -15000:1000:-2000);

%disp(logProbX(end))
%imagesc(reshape(mu(:,1),16,16));
%imagesc(reshape(vary(:,1),16,16));