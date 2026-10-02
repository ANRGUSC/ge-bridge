"""Exact residual loss for long retry chains, exponential kernel: lag-mD samples form an AR(1) sequence,
so P(all L+1 attempts bad) follows from a 1-D quadrature recursion."""
import numpy as np
from scipy.stats import norm
from gebridge import ge_params, k_ou
S=-1.5; m=8; Tc=1333.0; L=48
phi=np.exp(-m/Tc); sd=np.sqrt(1-phi**2)
x=np.linspace(-8,S,2601); dx=x[1]-x[0]; w=np.full_like(x,dx); w[0]=w[-1]=dx/2
K=norm.pdf((x[None,:]-phi*x[:,None])/sd)/sd; f=norm.pdf(x)
for _ in range(L): f=(f*w)@K
q=ge_params(S,k_ou(m,Tc))[0]
print("exact %.3e  lag GE %.3e"%((f*w).sum(), norm.cdf(S)*(1-q)**L))
