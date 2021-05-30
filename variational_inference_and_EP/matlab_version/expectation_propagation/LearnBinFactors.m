function [mu, sigma, pie, FF] = LearnBinFactors(X,K,iterations)
%% INPUT:
% K: number of latent factors
% X: observed data
%% Initialise mu, sigma, pie, ESS (lambda)
    [N,D] = size(X); 
    mu = rand(D,K);
    lambda = repmat(ones(1,K)*(1/K),N,1);
    FF = zeros(iterations,1); % store free energy
    sigma = 0.5;
    pie = ones(1,K) * (1/K);
    initial_Beta = rand(K,K,N);
    
    maxsteps = 35; % steps in EP iteration
    for iter = 1:iterations
        [lambda,F] = EP(X,mu,sigma,pie,lambda,maxsteps,initial_Beta);
        FF(iter) = F;
        iter,F
        ES = lambda;
        [ESS] = CalculateESS(ES);
        [mu, sigma, pie] = MStep(X,ES,ESS);    
    end
end
