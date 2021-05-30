function [lambda] = CalculateLambda(X,mu,sigma,pie,lambda,N,D,K)
    for i = 1:K            
        lambda(:,i) = Sigmoid(log(repmat(pie(i)/(1-pie(i)),N,1))...
                    - (1/sigma^2) *  (lambda * mu' + (0.5 - lambda(:,i)) * mu(:,i)' - X) * mu(:,i));   
    end
end