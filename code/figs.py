import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from gebridge import *
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,"legend.fontsize":6.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"lines.linewidth":1.3,"axes.linewidth":0.6,
 "grid.linewidth":0.4,"grid.color":"#d0d0d0","pdf.fonttype":42,"mathtext.fontset":"cm"})
BLUE,ORANGE,GRAY="#2a78d6","#eb6834","#6b6b6b"
rng=np.random.default_rng(2027)
KERNELS=[(k_se,BLUE,"Sq.-exp.","SE"),(k_ou,ORANGE,"Exponential (OU)","OU")]

def ETGE(sv,rho):
    q=norm.cdf(sv); p01,p10=ge_params(sv,rho); return q/p01+(1-q)/p10

def mc_ETGE(ker,T,sv,N,R=250,batch=25):
    """arXiv protocol: R sequences of length N; transition probabilities from observed counts
    (pooled over sequences) with Jeffreys 1/2 pseudo-count; 95% CI by bootstrap over sequences."""
    c=np.zeros((R,4))
    for b0 in range(0,R,batch):
        b=(sample_gaussian(ker,T,N,batch,rng)>=sv).astype(np.int8); x0,x1=b[:,:-1],b[:,1:]
        c[b0:b0+batch]=np.stack([((x0==0)&(x1==1)).sum(1),(x0==0).sum(1),((x0==1)&(x1==0)).sum(1),(x0==1).sum(1)],1)
    def est(cc):
        s=cc.sum(0); p01=(s[0]+.5)/(s[1]+1); p10=(s[2]+.5)/(s[3]+1); q=p10/(p01+p10); return q/p01+(1-q)/p10
    boot=[est(c[rng.integers(0,R,R)]) for _ in range(300)]
    return est(c), np.percentile(boot,2.5), np.percentile(boot,97.5)

# ---------- Fig 1: persistence scaling ----------
Tc=np.logspace(0,3,200)
fig,ax=plt.subplots(figsize=(3.45,2.5))
worst_small=worst_big=0
for ker,col,lab,_ in KERNELS:
    for sv,ls in [(0,"-"),(1,"--")]:
        ax.plot(Tc,ETGE(sv,ker(1,Tc)),ls,color=col,label=f"{lab}, $s={sv}$")
    for sv,mk in [(0,"o"),(1,"s")]:
        Ts=list(range(1,16))+[30,100,300,1000]; m_=[];lo=[];hi=[]
        for T in Ts:
            N=1200 if T<=15 else int(200*T)
            e,l,h=mc_ETGE(ker,T,sv,N); m_.append(e);lo.append(e-l);hi.append(h-e)
            dev=abs(e-ETGE(sv,ker(1,T)))/ETGE(sv,ker(1,T))
            if T<=15: worst_small=max(worst_small,dev)
            else: worst_big=max(worst_big,dev)
        ax.errorbar(Ts,m_,yerr=[lo,hi],fmt=mk,ms=3.0,mfc="white",mec=col,ecolor=col,elinewidth=0.7,capsize=1.2,mew=0.8)
print("Fig1 max rel dev: Tc<=15 %.4f ; Tc>=30 %.4f"%(worst_small,worst_big))
C=lambda sv: np.sqrt(np.pi)*(norm.cdf(sv)**2+(1-norm.cdf(sv))**2)/norm.pdf(sv)
x=np.logspace(0,3,60)
for sv in [0,1]:
    ax.plot(x,C(sv)*x,":",color=GRAY,lw=0.9,zorder=0); ax.plot(x,C(sv)*np.sqrt(x),":",color=GRAY,lw=0.9,zorder=0)
ax.text(260,260,r"$\propto T_c$",color=GRAY,fontsize=7.5); ax.text(300,18,r"$\propto \sqrt{D\,T_c}$",color=GRAY,fontsize=7.5)
ax.set_xscale("log");ax.set_yscale("log");ax.grid(True,which="major")
ax.set_xlabel(r"Normalized correlation time $T_c/D$");ax.set_ylabel(r"GE persistence time $\mathbb{E}[T_{\rm GE}]/D$")
h,l=ax.get_legend_handles_labels()
h+= [Line2D([],[],ls=":",color=GRAY,lw=0.9), Line2D([],[],ls="",marker="o",mfc="white",mec="k",ms=3.2)]
l+= ["Asymptotes (16), (17)","Monte Carlo, 95% CI"]
ax.legend(h,l,loc="upper left",frameon=False,handlelength=2.2)
fig.tight_layout(pad=0.3); fig.savefig("../fig_scaling.pdf"); fig.savefig("../fig_scaling.png",dpi=200)

# ---------- Fig 2: ARQ residual loss ----------
S,m,L=-1.5,8,3
d=json.load(open("arq.json"))
fig,axs=plt.subplots(2,1,figsize=(3.45,3.9),sharex=True)
for ax,(ker,col,lab,key),tag in zip(axs,KERNELS,["(a)","(b)"]):
    r=d[key]; T=[x["Tc"] for x in r]
    ax.axvspan(1e3,1e4,color="#eeeeee",zorder=0,lw=0); ax.text(3.2e3,1.2e-4,"shadowing\nregime",fontsize=7,ha="center",va="bottom",color="#555555")
    ax.plot(T,[x["truth"] for x in r],"-",color="k",lw=1.0,label="Gaussian (numerical)")
    ax.plot(T,[x["lagGE"] for x in r],"-",color=BLUE,lw=1.6,label="Lag-calibrated GE (Thm. 1)")
    ax.plot(T,[x["slotGE"] for x in r],"--",color=ORANGE,lw=1.6,label="Slot-calibrated GE")
    ax.plot(T,[x["iid"] for x in r],":",color=GRAY,lw=1.6,label="i.i.d.")
    Tm=[10,30,100,300,1000]; mc=[]
    for Tc_ in Tm:
        N=int(max(4000,200*Tc_)); x=sample_gaussian(ker,Tc_,N,250,rng)<S
        idx=np.arange(0,N-L*m); ab=np.ones((250,len(idx)),bool)
        for k in range(L+1): ab&=x[:,idx+k*m]
        mc.append(ab.mean())
    print(key,"MC",dict(zip(Tm,[f"{v:.3e}" for v in mc])))
    ax.plot(Tm,mc,"o",ms=4.2,mfc="white",mec="k",mew=0.9,label="Monte Carlo")
    ax.set_xscale("log");ax.set_yscale("log");ax.grid(True);ax.set_xticks([1,10,100,1000,10000])
    ax.text(0.02,0.95,f"{tag} {lab} kernel",transform=ax.transAxes,fontsize=8.5,va="top")
    ax.set_ylabel(r"Residual loss $P_{\mathrm{res}}$",fontsize=8.5); ax.set_ylim(1e-5,0.3); ax.tick_params(labelsize=8)
axs[1].set_xlabel(r"Normalized correlation time $T_c/D$",fontsize=8.5)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(0.54,1.0),columnspacing=1.0,handlelength=2.0)
fig.tight_layout(pad=0.3,h_pad=0.4,rect=(0,0,1,0.875)); fig.savefig("../fig_arq.pdf"); fig.savefig("../fig_arq.png",dpi=200)
