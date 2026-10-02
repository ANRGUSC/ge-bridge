import numpy as np
from scipy.stats import norm
from gebridge import ge_params
sig=8.0; F=12.0; s=-F/sig
print('pi0',norm.cdf(s))
for name,v,dc in [("Pedestrian",1.5,20),("Urban",15,20),("Highway",30,50)]:
  for D in [1e-3,0.125e-3]:
    rho=np.exp(-v*D/dc); p01,p10=ge_params(s,rho)
    print(f"{name:10s} v={v} dc={dc} D={D*1e3}ms Tc/D={dc/v/D:.0f} 1-rho={1-rho:.1e} p10={p10:.1e} p01={p01:.1e} D/p01={D/p01*1e3:.0f}ms D/p10={D/p10*1e3:.0f}ms")
