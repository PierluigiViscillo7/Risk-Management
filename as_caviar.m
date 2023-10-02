function AS_CAViaR=as_caviar(param,ret,alpha)

%% INPUT
% Param: vector of parameters
% ret: time-series of returns
% alpha: significance level

%% OUTPUT
% AS_CAViaR: AS-CAViaR level

b0=param(1);
b1=param(2);
b2=param(3);
b3=param(4);

T=length(ret);
qt=nan(T,1);
qt(1)=quantile(ret,alpha);

est_beta=nan(T,1);
est_beta(1)=((ret(1)-qt(1))*(alpha-(ret(1)<=qt(1))));

for i=2:T
    qt(i)=b0+b1*qt(i-1)+b2*max(ret(i-1),0)+b3*(-min(ret(i-1),0));
    est_beta(i)=((ret(i)-qt(i))*(alpha-(ret(i)<=qt(i))));
end

AS_CAViaR=mean(est_beta);