"""Closed-form Gaussian-fading -> Gilbert-Elliott bridge, plus simulation utilities.

Convention (matches the arXiv paper): slot n is BAD (0) if X(nD) < S, GOOD (1) otherwise.
a = (S - mu)/sigma is the normalized threshold (a = -M/sigma_dB for fade margin M dB).
p01 = P(bad -> good), p10 = P(good -> bad).
"""
import numpy as np
from scipy.special import owens_t
from scipy.stats import norm


def ge_params(a, rho):
    """Exact GE transition probabilities for threshold a and one-step correlation rho."""
    rho = np.asarray(rho, dtype=float)
    alpha = np.sqrt((1.0 - rho) / (1.0 + rho))
    t2 = 2.0 * owens_t(a, alpha)          # = P(X0 < a, X1 >= a)
    piB = norm.cdf(a)
    return t2 / piB, t2 / (1.0 - piB)      # p01, p10


def ge_params_mean_threshold(rho):
    v = 0.5 - np.arcsin(rho) / np.pi
    return v, v


def mean_bad_run_asymptotic(a, c, alpha_hoelder, Tc_over_D):
    """Mean bad-run length (slots) when 1 - rho ~ c (D/Tc)^alpha."""
    return norm.cdf(a) * np.sqrt(np.pi) / (norm.pdf(a) * np.sqrt(c)) * Tc_over_D ** (alpha_hoelder / 2)


# ---------------- kernels (normalized, D = 1) ----------------
def k_se(tau, Tc):
    return np.exp(-(np.asarray(tau, float) / Tc) ** 2)


def k_ou(tau, Tc):
    return np.exp(-np.abs(np.asarray(tau, float)) / Tc)


# ---------------- simulation ----------------
def sample_gaussian(kernel, Tc, N, n_paths, rng):
    """Stationary unit-variance Gaussian paths on the integer grid via circulant embedding."""
    if kernel is k_ou:  # exact AR(1)
        from scipy.signal import lfilter
        r = np.exp(-1.0 / Tc)
        e = rng.standard_normal((n_paths, N)) * np.sqrt(1 - r * r)
        e[:, 0] = rng.standard_normal(n_paths)
        return lfilter([1.0], [1.0, -r], e, axis=1)
    M = 2 * N
    lags = np.minimum(np.arange(M), M - np.arange(M))
    c = kernel(lags, Tc)
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0, None)   # SE: negative eigenvalues are O(1e-15) round-off
    out = np.empty((n_paths, N))
    for i in range(0, n_paths, 2):
        z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        y = np.fft.fft(np.sqrt(lam / M) * z)
        out[i] = y.real[:N]
        if i + 1 < n_paths:
            out[i + 1] = y.imag[:N]
    return out


def run_lengths(b, state):
    """Lengths of maximal runs of `state` in each row of binary array b (edge runs dropped)."""
    res = []
    for row in b:
        s = (row == state).astype(np.int8)
        d = np.diff(np.concatenate(([0], s, [0])))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        L = en - st
        keep = (st > 0) & (en < len(row))
        res.append(L[keep])
    return np.concatenate(res)
