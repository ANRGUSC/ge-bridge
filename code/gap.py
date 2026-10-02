import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
from gebridge import *
rng=np.random.default_rng(3)
def gap(ker,Tc,a):
    r1,r2=ker(1,Tc),ker(2,Tc)
    C=np.array([[1,r1,r2],[r1,1,r1],[r2,r1,1]])+1e-10*np.eye(3)
    pbbb=mvn(np.zeros(3),C,allow_singular=True).cdf([a]*3)
    pb=norm.cdf(a); pbb=mvn([0,0],[[1,r1],[r1,1]]).cdf([a,a])
    # conditional probs of bad at n+1
    one = pbb/pb                       # P(b|b)
    c_bb = pbbb/pbb                    # P(b|b,b)
    c_gb = (pbb-pbbb)/(pb-pbb)         # P(b|g_{n-1}, b_n)
    return max(abs(c_bb-one),abs(c_gb-one))
def tv(ker,Tc,a):
    x=sample_gaussian(ker,Tc,2**16,16,rng); L=run_lengths((x>=a).astype(int),0)
    p01,_=ge_params(a,ker(1,Tc)); mx=L.max()+1
    emp=np.bincount(L,minlength=mx)[1:]/len(L); k=np.arange(1,mx)
    geo=p01*(1-p01)**(k-1); tail=(1-p01)**(mx-1)
    return 0.5*(np.abs(emp-geo).sum()+tail), L.mean(), 1/p01
if __name__=="__main__":
 for a in [0,-1]:
   for ker,name in [(k_se,"SE"),(k_ou,"OU")]:
     for Tc in [2,5,10,20,50]:
         t,mL,th=tv(ker,Tc,a)
         print(f"a={a} {name} Tc={Tc:3} gap={gap(ker,Tc,a):.4f} TV={t:.3f} meanL sim={mL:.2f} th={th:.2f}")
