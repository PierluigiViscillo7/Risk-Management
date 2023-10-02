function [l,sigma2]=loglik_GARCHt(par,r)
% MATLAB function to specify the log-likelihood of the GARCH 
% model with t-student distribution.
% INPUTS 
%   r: Tx1 vector of the time series
%   Par: column vector with all the parameters
%   
%
% OUTPUTS
%   L: vector of the log likelihood of the GARCH(1,1) model
%   sigma2: vector of the conditional variance of GARCH(1,1) model
%

% GARCH parameters 
omega=par(1);
alpha=par(2);
beta=par(3);

v=par(4); % degrees of freedom of the t-student distribution

T=size(r,1);
l=zeros(T,1);
sigma2=zeros(T,1);
sigma2(1)=omega/(1-alpha-beta);

for i=2:T
    sigma2(i)=omega+alpha*r(i-1)^2+beta*sigma2(i-1);
    l(i)=gammaln((v+1)/2)-0.5*log(pi*(v-2))-gammaln(v/2)-0.5*log(sigma2(i))-0.5*(v+1)*log(1+(r(i)^2)/(sigma2(i)*(v-2)));
end



