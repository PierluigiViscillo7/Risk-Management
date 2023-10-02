function SAV_CAViaR=caviar_sav(param, ret, alpha) 
  
  % Function to compute the beta estimates of the Symmetric Absolute Value CAViaR
  %(CAViaR-SAV) of Engle and Manganelli(2004)
  % using regression quantiles, introduced by Koenker and Bassett(1978)
  b0= param(1);
  b1= param(2);
  b2= param(3);
  T=length(ret);
  qt=nan(T,1);
  qt(1)= quantile(ret, alpha);
  est_beta=nan(T,1);
  est_beta(1)= ((ret(1)- qt(1))* (alpha-(ret(1) <= qt(1))));
 
  for i=2:T
      qt(i)=b0 + b1 * qt(i - 1) + b2 * abs(ret(i - 1));
      est_beta(i)= ((ret(i)- qt(i))* (alpha-(ret(i) <= qt(i))));
  end
  
  SAV_CAViaR=mean(est_beta);