function [beta,stderr,vc,logl,exitflag]=Max_lik(lik_fct,b0,vc_type,A,b,Aeq,beq,lb,ub,nonlcon,options,varargin)
 %PURPOSE: computes the maximum-likelihood estimates and associated standard errors
 %         allows for Sandwiched standard errors (White) or usual hessian ones
 %         implementation by Michael Rockinger March 20, 2004.(it is possible that
 %         for large parameter values the Hessian behaves ugly)
 %-------------------------------------------------------------------------  
 %USAGE : [beta,stderr,vc,logl]=Max_lik(lik_fct,b0,vc_type,A,b,Aeq,beq,lb,ub,nonlcon,options,varargin);
 %-------------------------------------------------------------------------
 %INPUT :lik_fct - function, the likelihood function
 %       b0 - vector, parameters' initial values
 %       vc_type - instructions which variance-covariance matrix to implement.
 %                 if vc_type='Sandwich' white method will be used, else-
 %                 numerical hessian method.
 %      
 %       The following inputs are dedicated to the fmincon function. Please
 %       read the MATLAB help file for fmincon for a detailed description.
 %       A - Vector, constrains for the minimization
 %       b - Vector, constrains for the minimization
 %       Aeq - Vector, constrains for the minimization
 %       beq - Vector, constrains for the minimization
 %       lb - Vector, lower bound for the parameters
 %       ub - Vector, upper bound for the parameters
 %       nonlcon - subjects the minimization to the nonlinear inequalities
 %       options - function's options
 %       varargin - input arguments that are needed to calculate the likelihood function 
 %-------------------------------------------------------------------------
 %RETURN : beta - Vector, the parameters that maximize likelihood function
 %         stderr - Vector,  associated standard errors. These standard errors get computed
 %                  either via numerical hessians or via the sandwich (white) method.
 %         vc - Matrix, variance covariance matrix. calculation method as
 %              the one of the errors.
 %         logl - Scalar, the log-likelihood of the problem.
 %-------------------------------------------------------------------------
f0=feval(lik_fct,b0,varargin{:});
T=size(f0,1);

if T==1
    error('Likelihood function should return a column vector of loglikelihoods')
end

[beta,fval,exitflag] = fmincon(@ml_avg,b0,A,b,Aeq,beq,lb,ub,nonlcon,options,lik_fct,varargin{:});

hessian=HessMp(@ml_avg,beta,lik_fct,varargin{:})*T; % E_N(L_{theta,theta}(hat(theta))). \hat(H(theta_hat))

inv_h=-hessian\eye(size(hessian,1)); % -E_N(L_{theta,theta}(hat(theta)))^{-1}

g  = gradp(lik_fct,beta,varargin{:}); % L_{theta}(hat(theta)) (this is NxK matrix)

if strcmpi(vc_type,'Outer');
    
    disp('estimation of variance-covariance matrix via Outer Product');
    vc = inv(g'*g);  % OPG (outer-product of the gradients) 
 
elseif strcmpi(vc_type,'Sandwich')
    
    disp('estimation of variance-covariance matrix via Sandwich [White]');
    g  = gradp(lik_fct,beta,varargin{:});
    vc = inv_h*(g'*g)*inv_h;
    
else %default is information matrix
    
    disp('estimation of variance-covariance matrix via Hessian'); 
    vc = -inv_h;
    
end

if ~any(isreal(vc))
    disp('VC return not real. Trying estimation of variance-covariance matrix via Outer Product');
    vc = inv(g'*g);
end
stderr  = sqrt(diag(vc));
logl    = -fval*T; % sum(Loglik)=logl (we don't divide by T)
ztest   = beta./stderr; 
parno   = (1:size(beta,1))';

if T-size(beta,1)>0;
     pvalue  =  2*(1-normcdf( abs(ztest)));
 else
     pvalue = NaN*zeros(size(beta));
end

Res     = [ parno beta stderr ztest pvalue];

fprintf('\n\n\n **********************************************************************\n');
if T-size(beta,1)<=0;
    fprintf('\nWarning\n')
    fprintf('Model contains more parameters than observations \n')
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n')
end
fprintf('Number of observations: %12.4f\n',T);
fprintf('Value of likelihood     %12.4f\n',logl);
fprintf('Number of parameters    %12.4f\n',size(beta,1));
fprintf(' **********************************************************************\n');
fprintf('       parameter       beta        stderr    z-test      p-value\n');
fprintf('  %12.0f %12.4f  %12.4f %12.4f %12.4f\n', Res' );




function l=ml_avg(b,lik_fct,varargin);
l=feval(lik_fct,b,varargin{:});
l=-mean(l); % -E_N(L_i(theta))=1/N sum(L_i(theta))

function g=gradp(f,x0,varargin)
% computes the gradient of f evaluated at x
% uses forward gradients. Adjusts for possible differently scaled x by taking percentage increments
% this function is the equivalent to the gradp function of Gauss
% f should return either a scalar or a column vector
% x0 should be a column vector of parameters
f0=feval(f,x0,varargin{:}); 
[T,c]=size(f0);

if size(x0,2)>size(x0,1)
    x0=x0';
end
k=size(x0,1); % number of parameters wrt which one should compute gradient

h=0.00001; %some small number

g=zeros(T,k); %will contain the gradient
e=eye(k); 
for j=1:k;
    if x0(j)>1; % if argument is big enough, compute relative number   
        f1=feval(f,(x0.*( ones(k,1) +  e(:,j) *h )),varargin{:});    
        g(:,j)=(f1-f0)/(x0(j)*h);    
    else
        f1=feval(f, x0 +  e(:,j) *h ,varargin{:});    
        g(:,j)=(f1-f0)/h;    
    
    end
    
end

function H=HessMp(f,x0,varargin)
% computes the Hessian matrix of f evaluated at x0. If x0 has K elements,
% function returns a KxK matrix. The function f, given the ML context, is
% expected to be a column vector
% uses central differences 

% f should return either a scalar or a column vector
% x0 should be a column vector of parameters
f0=feval(f,x0,varargin{:}); 
[T,co]=size(f0);
if co>1; error('Error in HessMp, The function should be a column vector or a scalar'); end

[k,c]=size(x0);
if k<c,
    x0=x0';
end
k=size(x0,1); % number of parameters wrt which one should compute gradient

h=0.0001; %some small number

H=zeros(k,k); %will contain the Hessian
e=eye(k); 

h2=h/2;
for ii=1:k;
      if x0(ii)>100; % if argument is big enough, compute relative number   
        x0P= x0.*( ones(k,1) +  e(:,ii) *h2 );
        x0N= x0.*( ones(k,1) -  e(:,ii) *h2 );
        Deltaii = x0(ii)*h;
    else
        x0P = x0 +  e(:,ii) *h2;
        x0N = x0 -  e(:,ii) *h2;
        Deltaii = h;
    end
    
    for jj=1:ii
    if x0(jj)>100; % if argument is big enough, compute relative number   
        x0PP = x0P .* ( ones(k,1) +  e(:,jj) *h2 );
        x0PN = x0P .* ( ones(k,1) -  e(:,jj) *h2 );
        x0NP = x0N .* ( ones(k,1) +  e(:,jj) *h2 );
        x0NN = x0N .* ( ones(k,1) -  e(:,jj) *h2 );
        Delta = Deltaii*x0(jj)*h;
    else
        x0PP = x0P  +  e(:,jj) *h2; 
        x0PN = x0P  -  e(:,jj) *h2; 
        x0NP = x0N  +  e(:,jj) *h2; 
        x0NN = x0N  -  e(:,jj) *h2; 
        Delta = Deltaii*h;
    end
    
        fPP = feval(f,x0PP,varargin{:});   % forward,forward
        fPN = feval(f,x0PN,varargin{:});   % forward,backward
        fNP = feval(f,x0NP,varargin{:});    % backward,forward
        fNN = feval(f,x0NN,varargin{:});    % backward,backward
        
        H(ii,jj)=(sum(fPP)-sum(fPN)-sum(fNP)+sum(fNN))/Delta;
        H(jj,ii)=H(ii,jj);
    end
end