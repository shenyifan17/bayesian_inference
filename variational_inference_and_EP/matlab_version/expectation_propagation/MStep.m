% [mu, sigma, pie] = MStep(X,ES,ESS)
% 
% Inputs:
% -----------------
%        X NxD data matrix
%       ES NxK E_q[s]
%      ESS KxK sum over data points of E_q[ss'] (NxKxK)
%              if E_q[ss'] is provided, the sum over N is done for you.
% 
% Outputs:
% --------
%       mu DxK matrix of means in p(y|{s_i},mu,sigma)
%    sigma 1x1 standard deviation in same
%      pie 1xK vector of parameters specifying generative distribution for s
% 

function [mu, sigma, pie] = MStep(X,ES,ESS)

[N,D] = size(X);
if (size(ES,1)~=N), error('ES must have the same number of rows as X'); end
K = size(ES,2);
if (isequal(size(ESS),[N,K,K])), ESS = shiftdim(sum(ESS,1),1); end
if (~isequal(size(ESS),[K,K]))
	error('ESS must be square and have the same number of columns as ES');
end

mu = (inv(ESS)*ES'*X)';
sigma = sqrt((trace(X'*X)+trace(mu'*mu*ESS)-2*trace(ES'*X*mu))/(N*D));
pie = mean(ES,1);