function [pvalue,statistic]=Backtesting_VaR(VaR,alpha,ret,test)

% This function perfroms the three most udsed backtesting prodedures for
% VaR models:
% The Unconditional Coverage test (UC) (Kupiec,1995)
% The Independence test (IND) (Christoffersen,1998)
% The Conditional Coverage test (CC) (Christoffersen,1998)
%
% INPUTS:
% VaR: vector containing the VaR estimates
% alpha: confidence level VaR (usually 0.01 or 0.05)
% ret: vector of returns
% test: 'UC','IND','CC'
%
% OUTPUT 
% The model outputs the p-value and the test statistic of the choosen test

T=size(ret,1);
I=zeros(T,1); % Indicator sequence of violation of VaR

for j=1:T
    if ret(j)<=VaR(j)
        I(j)=1;
    else
        I(j)=0;
    end
end

% Unconditional Coverage test
n1=sum(I); % how many ones
n0=T-n1;   % how many zeros
alpha_hat=n1/T; % observed frequency of violation in the sample

LR_UC = -2*log(((1-alpha)^(n0) * alpha^n1)/((1-alpha_hat)^(n0)*alpha_hat^n1));


% Independence test 
N00=zeros(T-1,1);
N01=zeros(T-1,1);
N10=zeros(T-1,1);
N11=zeros(T-1,1);

for i=2:T
    if I(i)==0 && I(i-1)==0
        N00(i)=1;
    elseif I(i)==1 && I(i-1)==0
        N01(i)=1;
    elseif I(i)==0 && I(i-1)==1
        N10(i)=1;
    elseif I(i)==1 && I(i-1)==1
        N11(i)=1;
    end
end

% N(i,j) with i,j={0,1} is the number of observations with value 'i' followed
% by value 'j'
N00=sum(N00);
N01=sum(N01);
N10=sum(N10);
N11=sum(N11);

PI_01 = N01/(N01+N00);
PI_11 = N11/(N11+N10);
PI_2 = (N01 + N11)/(N00 + N01 + N10 + N11);
num=((1-PI_2)^(N00+N10))*PI_2^(N01+N11);
den=((1-PI_01)^N00)*(PI_01^N01)*((1-PI_11)^N10)*PI_11^N11;

LR_IND=-2*log(num/den);


% Conditional Coverage test 
LR_CC=LR_UC+LR_IND;


if strcmpi(test,'UC')
    pval_UC=1-chi2cdf(LR_UC,1);
    pvalue=pval_UC;
    statistic=LR_UC;
end

if strcmpi(test,'IND')
    pval_IND=1-chi2cdf(LR_IND,1);
    pvalue=pval_IND;
    statistic=LR_IND;
end

if strcmpi(test,'CC')
    pval_CC=1-chi2cdf(LR_CC,2);
    pvalue=pval_CC;
    statistic=LR_CC;
end





