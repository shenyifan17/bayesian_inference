function [ESS] = CalculateESS(ES)
    [N,K] = size(ES);
    ESS = zeros(K,K,N);
    for n = 1:N
        ESS(:,:,n) = ES(n,:)' * ES(n,:);  
        for k = 1:K
            ESS(k,k,n) = ES(n,k);
        end
    end    
    ESS = sum(ESS,3);
end