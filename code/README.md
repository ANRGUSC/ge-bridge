# Code for "From Gaussian Fading to Gilbert-Elliott: A Closed-Form Bridge for Link-Layer Modeling"

Requires Python 3 with numpy, scipy, matplotlib. Run each script from this folder.

| File | Produces |
|---|---|
| `gebridge.py` | Core library: Owen-T GE parameters (Theorem 1), kernels, Gaussian path samplers |
| `verify.py` | Sanity checks of Theorem 1, the arcsine identity and the asymptotes |
| `table.py` | Table I (Gudmundson shadowing examples, s = -1.5) |
| `arq2.py` | ARQ study (s = -1.5, m = 8, L = 3); writes `arq.json` and prints the error ranges quoted in Sec. VI |
| `dimK2.py` | Table II (minimum retry limit for residual loss <= 1e-3) |
| `longchain.py` | Long-retry-chain value (exponential kernel, Tc/D = 1333, L = 48) via AR(1) quadrature recursion |
| `gap.py` | Markov-gap helper (diagnostic of Sec. V) |
| `figs.py` | Fig. 1 (`fig_scaling.pdf`) and Fig. 2 (`fig_arq.pdf`); reads `arq.json`, writes to the parent folder |

`figs.py` runs the Monte Carlo for both figures and takes several minutes.
