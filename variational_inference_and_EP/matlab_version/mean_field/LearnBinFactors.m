function [mu, sigma, pie, FF] = LearnBinFactors(X,K,iterations)
%% INPUT:
% K: number of latent factors
% X: observed data
%% Initialise mu, sigma, pie, ESS (lambda)
    [N,D] = size(X); 
    mu = rand(D,K);
%     lambda = ones(N,K);
    lambda = repmat(ones(1,K)*(1/K),N,1);
    FF = zeros(iterations,1);
    sigma = 1;
    pie = ones(1,K) * (1/K);
    maxsteps = 500; % max 500 steps in Esteps, but usually finishes much early than 100
    for iter = 1:iterations
        [lambda,F] = MeanField(X,mu,sigma,pie,lambda,maxsteps);
        FF(iter) = F;
        % check whether Free Energy increases every time
%         if (iter > 1) && (F - FF(iter-1)<0)
%             disp('Free Energy decreases! Something is wrong!!!!')
%             disp(F)
%         else
%             disp(F)
%         end
        ES = lambda;
        [ESS] = CalculateESS(ES);
        [mu, sigma, pie] = MStep(X,ES,ESS);    
    end
end
