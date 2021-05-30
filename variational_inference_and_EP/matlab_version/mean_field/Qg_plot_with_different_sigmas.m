clc,clear,close all,clf
load best_run_K=8.mat
X_single = X(1,:);
lambda_single = ones(1,K);
maxsteps = 200;
%% use the best parameters learnt in Q(f) to plot F and log(F(t)-F(t-1))
[lambda,F,FF] = MeanField_with_plot(X_single,mu_best,sigma_best,pie_best,lambda_single,maxsteps);
% plot F
plot(FF)
title('Free Energy with the best parameters learnt')
xlabel('t (iterations)')
ylabel('Free Energy')
% plot log(F(t)-F(t-1))
log_diff = zeros(length(FF)-1,1);
for i = 1:length(FF) - 1
    log_diff(i) = log(FF(i + 1) - FF(i));
end
figure()
plot(log_diff)
title('log difference of Free Energy log(F(t) - F(t-1)')
xlabel('t (iterations)')
ylabel('log difference')

%% Plot F with three different sigmas:
[lambda_1,F_1,FF_1] = MeanField_with_plot(X_single,mu_best,0.01,pie_best,lambda_single,maxsteps);
[lambda_2,F_2,FF_2] = MeanField_with_plot(X_single,mu_best,0.1,pie_best,lambda_single,maxsteps);
[lambda_3,F_3,FF_3] = MeanField_with_plot(X_single,mu_best,1,pie_best,lambda_single,maxsteps);

figure()
plot(FF_1)
title('Sigma = 0.01')
xlabel('t (iterations)')
ylabel('Free Energy')

figure()
plot(FF_2)
title('Sigma = 0.1')
xlabel('t (iterations)')
ylabel('Free Energy')

figure()
plot(FF_3)
title('Sigma = 1')
xlabel('t (iterations)')
ylabel('Free Energy')








