clear, clc, clf, close all
genimages
X = Y; N = 400; D = 16; K = 8;
% run 100 times, choose the best free energy
times = 100;
iterations = 60;
mu_store = zeros(D,K,times);
sigma_store = zeros(times,1);
pie_store = zeros(1,K,times);
FF_store = zeros(iterations,times);
for t = 1:times
    disp(t)
    [mu, sigma, pie,FF] = LearnBinFactors(X,K,iterations);
    mu_store(:,:,t) = mu;
    sigma_store(t,1) = sigma;
    pie_store(:,:,t) = pie;
    FF_store(:,t) = FF;
    disp(FF(iterations))
end
    
% Find the one with the best Free energy:
[~,index] = max(FF_store(iterations,:));
mu_best = mu_store(:,:,index);
pie_best = pie_store(:,:,index);
FF_best = FF_store(:,index);
sigma_best = sigma_store(index);

% Plot a K images 
set(gcf,'Color',[0.9 0.9 0.9]); % Background color
colormap gray;
k=0;
nrows=3;
for i=1:K
    k=k+1;
    subplot(nrows,nrows,k);
    imagesc(reshape(mu_best(:,k),4,4),[0 2]);
    axis off;
end 

figure()
plot(FF_best)