function [F] = CalculateFreeEnergy(X,mu,sigma,pie,lambda,N,D,K)
    
    ES = lambda;
    ESS = CalculateESS(ES);
    F = - 0.5 * N * D * log(2*pi) - N * D * log(sigma) ... % constant
        + sum(sum(NanToZero(log(repmat(pie,N,1)).* lambda + log(repmat((1-pie),N,1)) .* (1-lambda))))...
        - sum(sum(NanToZero(lambda .* log(lambda) - (1-lambda) .* log(1-lambda)))) ... % entrophy
        - 0.5 * sigma^(-2) * (trace(X'*X)+trace(mu'*mu*ESS)-2*trace(ES'*X*mu)); % same with MStep caluclation
    % NOTE: we define a function NanToZero to remove nans created by
    % 0*log(0)
end