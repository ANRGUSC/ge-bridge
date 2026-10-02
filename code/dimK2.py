import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
from gebridge import *
S=-1.5; m=8; target=1e-3; pb=norm.cdf(S)
def exact(ker,Tc,L):
    j=L+1; idx=np.arange(j)*m; C=ker(idx[:,None]-idx[None,:],Tc)+1e-10*np.eye(j)
    return mvn(np.zeros(j),C,allow_singular=True,maxpts=200000*j,abseps=1e-8,releps=1e-4).cdf(np.full(j,S))
for ker,n in [(k_se,'SE'),(k_ou,'OU')]:
  for Tc in [5,10,20,40]:
    p01,p10=ge_params(S,ker(1,Tc)); mu=1-p01-p10; st=pb+(1-pb)*mu**m; q=ge_params(S,ker(m,Tc))[0]
    Li=int(np.ceil(np.log(target)/np.log(pb)-1)); Ls=int(np.ceil(np.log(target/pb)/np.log(st))); Ll=int(np.ceil(np.log(target/pb)/np.log(1-q)))
    Le=next(L for L in range(0,60) if exact(ker,Tc,L)<=target)
    print(n,Tc,'iid',Li,'slot',Ls,'lag',Ll,'exact',Le)
