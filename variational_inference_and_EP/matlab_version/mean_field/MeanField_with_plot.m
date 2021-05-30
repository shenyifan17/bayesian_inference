function [lambda,F,FF] = MeanField_with_plot(X,mu,sigma,pie,lambda,maxsteps)
%% OUTPUT: 
    % lambda -- 1*K 
    % F - scalar (Free energy, the lower bound)
%% INPUT:
    % X -- 1*D (observed data)
    % mu -- D*K (means)
    % pie -- 1*K (priors on s)
    % lambda0 -- N*K (initial values for lambda)
    % maxstep -- scalar (maximum number of steps)
    % sigma -- scalar
%% Code:
    [N,D] = size(X); K = length(pie);
    eplison = 1e-2;
%     FF = zeros(maxsteps,1);
    for iter = 1 : maxsteps
        lambda = CalculateLambda(X,mu,sigma,pie,lambda,N,D,K);
        F = CalculateFreeEnergy(X,mu,sigma,pie,lambda,N,D,K);
        FF(iter) = F; 
        if (iter>10) && (abs(FF(iter-1) - F) < eplison )
            break
            
        end        
    end
end
