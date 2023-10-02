%%
clear 
clc
%%
% Daily historical prices of the NASDAQ Composite index: 
%January 1, 2010 – September 1, 2022
data=readtable("NASDAQ.csv");

dates=datetime(table2array(data(:,1)));
NASDAQ=table2array(data(:,5));
r=diff(log(NASDAQ)); % return series
T=size(r,1);
%% GARCH(1,1)
% Using MATLAB biult in function
mdl_garch=garch('GARCHLags',1,'ARCHLags',1,'Distribution','t');

garchmdl=estimate(mdl_garch,r);
[v_GARCH,~]=infer(garchmdl,r);

z_garch=r./v_GARCH;
[~,pval_lbqtest_garch_r]=lbqtest(z_garch,'Lags',[1,5,10]); 
[~,pval_lbqtest_garch_r2]=lbqtest(z_garch.^2,'Lags',[1,5,10]);


% Using Maximum Likelihood
options=optimset('Display','iter','Diagnostics','on');
b0=[0.002;0.55;0.1;3];
lb=[eps;eps;eps;2];
ub=[inf;1;1;50];

[par_GARCH,~,~,logl_GARCH,~]=Max_lik('loglik_GARCHt',b0,'Sandwich',[],[],[],[],lb,ub,[],options,r);
[~,sigma2_garch]=loglik_GARCHt(par_GARCH,r);

z_garchL=r./sigma2_garch;
[~,pval_lbqtest_garch_rL]=lbqtest(z_garchL,'Lags',[1,5,10]); 
[~,pval_lbqtest_garch_r2L]=lbqtest(z_garchL.^2,'Lags',[1,5,10]);

%% GJR-GARCH(1,1)

% using MATLAB built in function
mdl_gjr=gjr('GARCHLags',1,'ARCHLags',1,'LeverageLags',1,'Distribution','t');

gjrmdl=estimate(mdl_gjr,r);
[v_gjr,~]=infer(gjrmdl,r);

z_gjr=r./v_gjr;
[~,pval_lbqtest_gjr_r]=lbqtest(z_gjr,'Lags',[1,5,10]); 
[~,pval_lbqtest_gjr_r2]=lbqtest(z_gjr.^2,'Lags',[1,5,10]);

% using maxlik
b0=[0.002;0.55;0.1;0.1;3];
lb=[eps;eps;eps;eps;2];
ub=[inf;1;1;1;50];

[par_gjr,stderr,vc,logl_gjr,exitflag]=Max_lik('loglik_GJR_GARCHt',b0,'Sandwich',[],[],[],[],lb,ub,[],options,r);
[~,sigma2_gjr]=loglik_GJR_GARCHt(par_gjr,r);

z_gjrL=r./sigma2_gjr;
[~,pval_lbqtest_gjr_rL]=lbqtest(z_gjrL,'Lags',[1,5,10]); 
[~,pval_lbqtest_gjr_r2L]=lbqtest(z_gjrL.^2,'Lags',[1,5,10]);
%% Setting the alpha for the VaR equal to 5%
alpha=0.05;

%% VaR Garch
VaR_garch=tinv(alpha,par_GARCH(4)).*sqrt(sigma2_garch);

figure
plot(dates(2:end),r)
hold on
plot(dates(2:end),VaR_garch)
hold off
title('95% VaR GARCH(1,1)')
legend('retruns','95% VaR',Location='best')
%% VaR GJR-Garch
VaR_gjr=tinv(alpha,par_gjr(5)).*sqrt(sigma2_gjr);

figure
plot(dates(2:end),[r,VaR_gjr])
title('95% VaR GJR-GARCH')
legend('retruns','95% VaR',Location='best')

%% CaViar Asymmetric Slope
x0=[0.2;0.5;0.3;0.6];
[beta_as]=fminsearch('caviar_as',x0,options,r,alpha);

qt_CaViaR_AS=nan(T,1);
qt_CaViaR_AS(1)=quantile(r,alpha);

for i=2:T
    qt_CaViaR_AS(i)=beta_as(1)+beta_as(2)*qt_CaViaR_AS(i-1)+beta_as(3)*max(r(i-1),0)-beta_as(4)*min(r(i-1),0);
end

figure
plot(dates(2:end),[r qt_CaViaR_AS])
title('95% VaR CaViaR Asymmetric Slope')
legend('retruns','95% VaR',Location='best')

%% CaViar Indirect GARCH(1,1)

x0=[0.002;0.1;0.3];
[beta_ig]=fminsearch('caviar_ig',x0,options,r,alpha);

qt_CaViaR_IG=nan(T,1);
qt_CaViaR_IG(1)=quantile(r,alpha);

for i=2:T
    qt_CaViaR_IG(i)=-sqrt(beta_ig(1)+beta_ig(2)*qt_CaViaR_IG(i-1).^2+beta_ig(3)*r(i-1)^2);
end

figure
plot(dates(2:end),[r qt_CaViaR_IG])
title('95% VaR CaViaR Indirect Garch')
legend('retruns','95% VaR',Location='best')

%% Backtesing VaR models

VaR=[VaR_garch,VaR_gjr,qt_CaViaR_AS,qt_CaViaR_IG];

pvalue=zeros(4,3);
tstat=zeros(4,3);
test={'UC','IND','CC'};

for i=1:4
    for j=1:3
        [pvalue(i,j),tstat(i,j)]=Backtesting_VaR(VaR(:,i),alpha,r,test(1,j));
    end
end

pvalue=pvalue';
tstat=tstat';
%% Collect all information into a table
pval_garch=pvalue(:,1);
pval_gjr=pvalue(:,2);
pval_AS=pvalue(:,3);
pval_IG=pvalue(:,4);

TAB=table(pval_garch,pval_gjr,pval_AS,pval_IG, ...
    VariableNames={'p-value GARCH';'p-value GJR-GARCH';'p-value CaViaR AS';'p-value CaViaR IG'}, ...
    RowNames={'UC','IND','CC'});




