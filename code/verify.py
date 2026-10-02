import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
from gebridge import *
rng=np.random.default_rng(1)
# 1) Owen-T formula vs bivariate CDF
worst=0
for a in [-2,-1,-0.5,0,0.5,1.5]:
  for rho in [-0.5,0.1,0.5,0.9,0.99,0.999]:
    p01,p10=ge_params(a,rho)
    P=mvn(mean=[0,0],cov=[[1,rho],[rho,1]]).cdf([a,a])
    worst=max(worst,abs(p01-(norm.cdf(a)-P)/norm.cdf(a)))
print("max err Owen vs bvn:",worst)
print("arcsine check:",ge_params(0,0.7),ge_params_mean_threshold(0.7))
# 2) Asymptotic vs exact mean bad run 1/p01
for a in [0,-1]:
  for Tc in [10,100,1000]:
    se=1/ge_params(a,k_se(1,Tc))[0]; ou=1/ge_params(a,k_ou(1,Tc))[0]
    print(a,Tc,"SE",se,mean_bad_run_asymptotic(a,1,2,Tc),"OU",ou,mean_bad_run_asymptotic(a,1,1,Tc))
# 3) Rice check (SE, time units): D*meanrun -> Phi(a)*pi*sqrt2*Tc*exp(a^2/2)
a=-1;Tc=50;print("Rice",1/ge_params(a,k_se(1,Tc))[0], norm.cdf(a)*np.pi*np.sqrt(2)*Tc*np.exp(a*a/2))
# 4) Monte Carlo
for ker,name in [(k_se,"SE"),(k_ou,"OU")]:
  for Tc in [5,20]:
    x=sample_gaussian(ker,Tc,2**15,16,rng)
    print(name,Tc,"emp rho",np.mean(x[:,1:]*x[:,:-1]),"true",ker(1,Tc),"var",x.var())
    for a in [0,-1]:
      b=(x>=a).astype(int)
      L=run_lengths(b,0)
      print("   a",a,"mean bad run sim",L.mean(),"theory",1/ge_params(a,ker(1,Tc))[0])
