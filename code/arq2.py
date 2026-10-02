import numpy as np, json, sys
from scipy.stats import multivariate_normal as mvn, norm
from gebridge import *
S=-1.5; m=8; L=3
pb=norm.cdf(S)
def allbad(ker,Tc,j):
    if j==0: return 1.0
    if j==1: return pb
    idx=np.arange(j)*m; C=ker(idx[:,None]-idx[None,:],Tc)+1e-10*np.eye(j)
    return mvn(np.zeros(j),C,allow_singular=True,maxpts=100000*j,abseps=1e-9,releps=1e-5).cdf(np.full(j,S))
def preds(ker,Tc,LL=L):
    p01,p10=ge_params(S,ker(1,Tc)); mu=1-p01-p10; st=pb+(1-pb)*mu**m; q=ge_params(S,ker(m,Tc))[0]
    iid=[pb**j for j in range(LL+2)]; slot=[1.0]+[pb*st**(j-1) for j in range(1,LL+2)]; lag=[1.0]+[pb*(1-q)**(j-1) for j in range(1,LL+2)]
    return iid,slot,lag
def eta(seq,LL=L): return (1-seq[LL+1])/sum(seq[:LL+1])
Tcs=np.logspace(0,4,21); out={}
for ker,n in [(k_se,'SE'),(k_ou,'OU')]:
    rows=[]
    for Tc in Tcs:
        ex=[allbad(ker,Tc,j) for j in range(L+2)]; i,s_,l=preds(ker,Tc)
        rows.append(dict(Tc=float(Tc),truth=ex[L+1],iid=i[L+1],slotGE=s_[L+1],lagGE=l[L+1],eta_t=eta(ex),eta_i=eta(i),eta_s=eta(s_),eta_l=eta(l)))
    out[n]=rows
    t=np.array([r['truth'] for r in rows])
    for k in ['iid','slotGE','lagGE']:
        v=np.array([r[k] for r in rows])/t; print(n,k,'ratio min %.3f@%.1f max %.3f@%.1f'%(v.min(),Tcs[v.argmin()],v.max(),Tcs[v.argmax()]))
    print(n,'truth max %.4f'%t.max(),'iid %.2e'%rows[0]['iid'])
    print(n,'max|eta_l-eta_t| %.4f  max eta_i-eta_t %.3f  max|eta_s-eta_t| %.3f'%(max(abs(r['eta_l']-r['eta_t']) for r in rows),max(r['eta_i']-r['eta_t'] for r in rows),max(abs(r['eta_s']-r['eta_t']) for r in rows)))
    r100=[r for r in rows if abs(r['Tc']-100)<1][0]; print(n,'eta@100 exact %.3f iid %.3f lag %.3f'%(r100['eta_t'],r100['eta_i'],r100['eta_l']))
json.dump(out,open('arq.json','w'))
