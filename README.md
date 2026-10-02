# ge-bridge

**From Gaussian Fading to Gilbert–Elliott: A Closed-Form Bridge for Link-Layer Modeling**

Bhaskar Krishnamachari and Victor Gutierrez
Viterbi School of Engineering, University of Southern California

This repository contains the paper and the code used to produce all results and
figures for the above work.

> **Paper status:** in submission. A preprint is available on arXiv:
> **https://arxiv.org/pdf/2604.03160**

## Overview

Correlated Gaussian processes describe dB-domain shadow fading, while
Gilbert–Elliott (GE) models provide tractable binary channel abstractions for
link-layer analysis. This work derives **closed-form GE parameters** that exactly
match the one-step conditional probabilities of a thresholded stationary Gaussian
process:

- For arbitrary thresholds, the mapping uses Owen's *T*-function; at the mean
  threshold it reduces to a classical arcsine identity.
- The covariance kernel enters only through the correlation at the sampling lag.
- Distinct persistence-time scaling laws emerge: **linear** growth with the
  correlation time for the squared-exponential kernel and **square-root** growth
  for the exponential kernel at fixed slot duration.
- Applied to truncated ARQ, a GE model matched directly at the retransmission lag
  stays within a factor of 1.75 of the Gaussian benchmark, whereas independent
  losses underestimate residual loss by more than three orders of magnitude and a
  slot-level GE model misestimates it by up to a factor of 24.

## Repository layout

```
code/      Python scripts that reproduce every result, table, and figure
paper/     LaTeX source, bibliography, figures, and the compiled PDF
LICENSE    BSD 3-Clause License
```

## Code

Requires Python 3 with `numpy`, `scipy`, and `matplotlib`. Run each script from
the `code/` folder.

| File | Produces |
|---|---|
| `gebridge.py` | Core library: Owen-T GE parameters (Theorem 1), kernels, Gaussian path samplers |
| `verify.py` | Sanity checks of Theorem 1, the arcsine identity, and the asymptotes |
| `table.py` | Table I (Gudmundson shadowing examples, s = -1.5) |
| `arq2.py` | ARQ study (s = -1.5, m = 8, L = 3); writes `arq.json` and prints the error ranges quoted in the ARQ section |
| `dimK2.py` | Table II (minimum retry limit for residual loss <= 1e-3) |
| `longchain.py` | Long-retry-chain value (exponential kernel, Tc/D = 1333, L = 48) via AR(1) quadrature recursion |
| `gap.py` | Markov-gap helper (diagnostic) |
| `figs.py` | Fig. 1 (`fig_scaling.pdf`) and Fig. 2 (`fig_arq.pdf`); reads `arq.json`, writes to the parent folder |

`figs.py` runs the Monte Carlo for both figures and takes several minutes.

## Paper

The `paper/` folder contains the LaTeX source (`main.tex`, `refs.bib`),
figures, and the compiled PDF (`GE_bridge_paper.pdf`).

## License

Released under the BSD 3-Clause License. See [LICENSE](LICENSE).
