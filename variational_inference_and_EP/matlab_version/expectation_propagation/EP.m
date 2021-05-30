function [lambda,F] = EP(X,mu,sigma,pie,~,maxsteps,beta)
%% OUTPUT: 
    % lambda -- N*K 
    % F - scalar (Free energy, the lower bound)
%% INPUT:
    % X -- N*D (observed data)
    % mu -- D*K (means)
    % pie -- 1*K (priors on s)
    % maxstep -- scalar (maximum number of steps)
    % sigma -- scalar
%% Code:
[N,D] = size(X); K = length(pie);
% define theta outside to avoid for loop computation...
theta = zeros(N,K);

for n = 1:N
    theta(n,:) = (-1/(2*sigma^2)) * diag(mu'*mu)' + (1/sigma^2) * X(n,:) * mu ...
        + log(pie) - log(1 - pie);
end

for iter = 1 : maxsteps 
    for n = 1:N     % for each training data
        for i = 1:K     % for each pairwise factor (si,sj)
            for j = 1:K                
                w_ij =  (-1/(2*sigma^2)) * mu(:,i)' * mu(:,j);
%                 theta_i = (1/sigma^2) * X(n,:) * mu(:,i) - ... 
%                           (-1/(2*sigma^2)) *  mu(:,i)' * mu(:,i) + ... 
%                           log(pie(i)) - log(1-pie(i));
%                 theta_j = (1/sigma^2) * X(n,:) * mu(:,j) - ... 
%                           (-1/(2*sigma^2)) *  mu(:,j)' * mu(:,j) + ... 
%                           log(pie(j)) - log(1-pie(j));                       
                N_i = theta(n,i) + sum(beta(:,i,n)) - beta(i,i,n) - beta(j,i,n);
                N_j = theta(n,j) + sum(beta(:,j,n)) - beta(i,j,n) - beta(j,j,n);
                % updated beta:
%                 beta(j,i,n) = log((exp(w_ij+N_j)+1)/(exp(N_j)+1));
%                 beta(i,j,n) = log((exp(w_ij+N_i)+1)/(exp(N_i)+1));
                % update beta (with alpha included for convergence)
                 alpha = 0.6;
                beta(j,i,n) = (1-alpha)*beta(j,i,n) + alpha * log((exp(w_ij+N_j)+1)/(exp(N_j)+1));
                beta(i,j,n) = (1-alpha)*beta(j,i,n) + alpha *log((exp(w_ij+N_i)+1)/(exp(N_i)+1));
            end 
        end
    end 
end

% calculate lambda and free energy (outside four for loops)
lambda = zeros(N,K);
for n = 1:N
    lambda(n,:) = Sigmoid(theta(n,:) + sum(beta(:,:,n)));
end
F = CalculateFreeEnergy(X,mu,sigma,pie,lambda,N,D,K); 

end 
