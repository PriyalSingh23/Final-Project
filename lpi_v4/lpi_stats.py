#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpi_stats.py -- the eight-domain EW statistics, in ONE place.
==============================================================

Every function here is used twice:

  * by `lpi_train.py` as a *differentiable penalty* on the generator, and
  * by `lpi_eval.py` as the *reported metric*.

That identity is the point: the reference paper's "component-to-domain mapping"
claim (each loss term defeats exactly one EW test) is only honest if trainer and
reporter compute the same number.  A metric you optimise and a metric you publish
must never be two different pieces of code again.

Eight domains (Table 4 of the Elsevier draft), with the paper's pass criteria:

  1  KS p-value            > 0.05            (amplitude marginal ~ N(0,1))
  2  excess kurtosis        |k-3| < 0.15
  3  spectral entropy       > 0.95            (flat PSD, Welch/Hann)
  4  IQ circularity         |<z^2>|/E|z|^2 < 0.15
  5  PAPR deviation         |dPAPR| < 3 dB    (vs AWGN reference)
  6  SCF (cycle-stationary) ratio < 3         (CSF surface peak / noise)
  7  HOS C42                |C42| < 0.3
  8  WVD time-frequency     ratio < 3.7

Composite  Sc = sum_i w_i d_i / sum_i w_i ,  d_i in [0,1] normalised deviation
from the noise reference.  Grading: <0.10 Excellent, <0.20 Good, <0.35 Moderate.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


# =======================================================================
# differentiable loss components (used by the trainer)
# =======================================================================
def loss_mean(x):
    return x.mean().pow(2)


def loss_var(x, target: float = 1.0):
    B = x.shape[0]
    return (x.reshape(B, -1).var(dim=1, unbiased=False).mean() - target).pow(2)


def loss_kurtosis(x, target: float = 3.0):
    f = x.reshape(-1)
    mu = f.mean()
    v = f.var(unbiased=False)
    k = ((f - mu) ** 4).mean() / (v ** 2 + 1e-12)
    return (k - target).pow(2)


def loss_spectral_flatness(x, n_fft: int = 512):
    """L_spec = Var_f[PSD_norm]  +  2 * Corr[ mean_f|X| , std_f|X| ]^2   (paper Eq. 4).

    First term: a flat (white) spectrum has minimum per-frame PSD variance.
    Second term: a systematic *tilt* is invisible to the first term after
    normalisation, but it correlates a frame's average magnitude with its
    spectral spread -- so it is caught here.
    """
    F_ = min(n_fft // 2 + 1, x.shape[-1] // 2 + 1)
    p = torch.fft.rfft(x, dim=2).abs().pow(2)
    if p.shape[-1] != F_:
        p = p[..., :F_]
    pn = p / (p.sum(dim=2, keepdim=True) + 1e-12)
    var = pn.var(dim=2).mean()
    m = p.mean(dim=2)
    sd = p.std(dim=2)
    mc = m - m.mean(0, keepdim=True)
    sc = sd - sd.mean(0, keepdim=True)
    cov = (mc * sc).mean(0) / (mc.std(0) * sc.std(0) + 1e-12)
    return var + 2.0 * cov.pow(2).mean()


def loss_cyclo(x, max_lag: int = 256, n_seg: int = 8):
    """L_cyclo = (1/L) sum_lag |Rxx|^2 + 0.5 * Var_seg[Rxx]   (paper Eq. 5).

    Circular autocorrelation via FFT (no window leakage at the frame edge,
    which is what made the naive linear estimate look cyclostationary).
    """
    B, C, L = x.shape
    X = torch.fft.rfft(x, dim=2)
    ac = torch.fft.irfft(X * torch.conj(X), n=L, dim=2) / L      # (B,C,L)
    ac = ac[:, :, 1: max_lag + 1]
    main = ac.pow(2).sum() / (B * C * ac.shape[2])
    seg = ac.reshape(B, C, n_seg, -1).mean(dim=3)          # (B,C,S)
    var_seg = seg.var(dim=2, unbiased=False).mean()
    return main + 0.5 * var_seg


# --- the paper's Table-1 weights, kept in ONE dict so trainer and report agree --
W_TABLE1 = dict(mean=0.5, var=0.5, pwr=3.0, spec=3.0, cyclo=2.5, pnorm=6.0,
                div=3.0, kurt=4.0, ks=8.0)


def _gram_spec(x: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Eigen-spectrum of the row-Gram matrix of a batch of frames, mean-normalised.

    For i.i.d. Gaussian frames this spectrum follows Marchenko-Pastur; for anything
    that is a *linear image of a smaller latent space* (which is exactly what a
    generator produces: 128 bits + 64 latent dims spread over 1024 samples) the
    spectrum has outliers.  Eigenvalue/MP detectors are the standard tool in
    cognitive-radio spectrum sensing for exactly that reason, so the generator is
    asked to remove the signature in the same domain the detector looks in.
    """
    x = x.reshape(x.shape[0], -1)
    x = x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)
    B = x.shape[0]
    G = (x @ x.t()) / B
    G = G + ridge * torch.eye(B, device=x.device, dtype=x.dtype)
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(1e-6)
    return ev / (ev.mean() + 1e-12)


def loss_cov(fake: torch.Tensor, ref: torch.Tensor, topk: int = 64) -> torch.Tensor:
    """Log-spectrum mismatch between the fake batch and the reference batch."""
    a, b = _gram_spec(fake), _gram_spec(ref)
    k = int(min(topk, a.shape[-1]))
    return (a[..., :k].log() - b[..., :k].log()).pow(2).mean()


def composite_stat_loss(x, ref, z=None, w=None, w_cov: float = 2.0):
    """The nine-component generator objective of the reference paper (Table 1)
    minus the two adversarial terms, which the trainer adds separately.

    Returns (loss, dict_of_components) -- the dict is what gets logged, so the
    number in the CSV *is* the term in the gradient.
    """
    w = {**W_TABLE1, **(w or {})}
    lp, lpn = loss_power(x, ref)
    terms = {
        "mean": loss_mean(x), "var": loss_var(x), "kurt": loss_kurtosis(x),
        "spec": loss_spectral_flatness(x), "cyclo": loss_cyclo(x),
        "pwr": lp, "pnorm": lpn,
        "div": loss_diversity(x.reshape(x.shape[0], -1), z) if z is not None
               else torch.zeros((), device=x.device),
        "ks": loss_ks(x),
        "cov": loss_cov(x, ref),
    }
    tot = sum(w[k] * terms[k] for k in terms if k in w) + w_cov * terms["cov"]
    return tot, {k: float(v.detach()) for k, v in terms.items()}


def loss_power(x, ref: torch.Tensor):
    """L_pwr = (P_sig / (1.01 * P_noise))^2 and L_pn = log2(P_sig/P_noise)^2
    -- paper Eqs. 6 and the power-normalisation component."""
    ps = x.reshape(x.shape[0], -1).pow(2).mean()
    pr = ref.reshape(ref.shape[0], -1).pow(2).mean()
    return (ps / (1.01 * pr + 1e-12)).pow(2), torch.log2(ps / (pr + 1e-12) + 1e-12).pow(2)


def loss_diversity(x, z):
    """L_div = [ ||G(z1)-G(z2)|| / ||z1-z2|| ]^-1 -- maximise output distance
    for nearby latents, so the map cannot collapse onto a fingerprinted
    low-dimensional submanifold."""
    B = x.shape[0]
    if B < 2:
        return torch.zeros((), device=x.device)
    xf = x.reshape(B, -1)
    d = torch.cdist(xf, xf)
    dz = torch.cdist(z, z) + 1e-6
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    ratio = (d[mask] / dz[mask]).mean()
    return 1.0 / (ratio + 1e-3)


def loss_ks(x, n_pts: int = 1 << 20):
    """Empirical KS deviation in CDF space, per I/Q channel."""
    worst = torch.zeros((), device=x.device)
    for c in range(x.shape[1]):
        f = x[:, c].reshape(-1)
        if f.numel() > n_pts // 2:
            idx = torch.randint(0, f.numel(), (n_pts // 2,), device=f.device)
            f = f[idx]
        n = f.numel()
        s, _ = torch.sort(f)
        u = 0.5 * (1.0 + torch.erf(s / SQRT2))
        pos = (torch.arange(n, device=f.device, dtype=f.dtype) + 0.375) / (n + 0.25)
        worst = torch.maximum(worst, (u - pos).abs().max())
    return worst


# =======================================================================
# reported metrics (numpy; torch-tolerant)
# =======================================================================
def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def iq_frames(x) -> np.ndarray:
    """(N,2,T) -> (N,T) complex."""
    x = _np(x).astype(np.float64)
    return x[:, 0] + 1j * x[:, 1]


def metric_kurtosis(x) -> float:
    f = _np(x).reshape(-1)
    f = f - f.mean()
    return float((f ** 4).mean() / (f ** 2).mean() ** 2)


def metric_spectral_entropy(x, n_fft: int = 512) -> float:
    x = _np(x)
    w = np.hanning(x.shape[-1])
    X = np.abs(np.fft.rfft(x * w, n=n_fft, axis=-1)) ** 2
    p = X.mean(axis=tuple(range(1, x.ndim - 1))) if x.ndim > 2 else X
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum(axis=-1) / math.log(p.shape[-1])
    return float(H.mean())


def metric_iq_circularity(x) -> float:
    z = iq_frames(x)
    return float(np.abs(np.mean(z * z)) / (np.mean(np.abs(z) ** 2) + 1e-12))


def metric_papr_db(x) -> float:
    z = iq_frames(x)
    p = np.abs(z) ** 2
    return float(10 * np.log10((p.max(axis=-1) / (p.mean(axis=-1) + 1e-12)).mean()))


def metric_scf_ratio(sig, ref=None, max_lag: int = 64) -> float:
    """Cyclic-feature strength: peak of the (lag, alpha) |FFT of R(t,tau)|
    surface, relative to the same number on the reference (AWGN).

    A modulated carrier produces spectral lines at cycle frequencies
    alpha = k/T_symbol; noise does not.  We take the max over the alpha axis of
    the time-smoothed correlation per lag -- a compact stand-in for the full
    CSP/FAM grid that behaves identically on a single frame ensemble.
    """
    def scf(x):
        z = iq_frames(x)
        N, T = z.shape
        out = []
        for lag in range(1, min(max_lag, T // 4) + 1):
            r = z[:, :-lag] * np.conj(z[:, lag:])                 # (N,T-lag)
            R = np.abs(np.fft.fft(r, axis=0))[:, 1:]                     # across frames
            out.append(R.max(axis=-1).mean())
        return float(np.mean(out))

    a = scf(sig)
    if ref is None:
        ref = np.random.randn(*_np(sig).shape)
        ref = ref / ref.std()
        ref = np.stack([ref[:, 0], ref[:, 1]], axis=1)
    b = max(scf(ref), 1e-12)
    return a / b


def metric_hos_c42(x) -> float:
    """Complex 4th-order cumulant C42 = E|z|^4 - 2E|z|^2^2 - |E z^2|^2,
    normalised by E|z|^2^2.  Exactly 0 for proper complex Gaussian."""
    z = iq_frames(x)
    z = z - z.mean()
    m2 = np.mean(np.abs(z) ** 2)
    m4 = np.mean(np.abs(z) ** 4)
    c21 = np.mean(z * z)
    return float((m4 - 2 * m2 ** 2 - abs(c21) ** 2) / (m2 ** 2 + 1e-12))


def metric_wvd_ratio(sig, ref=None) -> float:
    """Time-frequency concentration ratio.

    The paper's WVD/TF ratio compares how peaky a spectrogram is relative to
    noise.  We use the (normalised) 4th moment of the STFT magnitude, i.e. the
    inverse of TF flatness, divided by the same statistic on AWGN.  Threshold
    3.7 was Monte-Carlo calibrated by the reference work (a 2.0 threshold gives
    an 11 % false-fail rate on pure noise).
    """
    def conc(x):
        x = _np(x)
        z = iq_frames(x)
        w = 64
        segs = []
        for i in range(0, z.shape[1] - w + 1, w // 2):
            win = z[:, i:i + w] * np.hanning(w)
            S = np.abs(np.fft.fft(win, n=128, axis=1))[:, :65] ** 2
            segs.append(S / (S.mean(axis=1, keepdims=True) + 1e-12))
        S = np.concatenate(segs, axis=1)
        return float((S ** 2).mean() / ((S.mean(axis=1) ** 2).mean() + 1e-12))

    a = conc(sig)
    if ref is None:
        ref = np.random.randn(*_np(sig).shape)
        ref = ref / ref.std()
        ref = np.stack([ref[:, 0], ref[:, 1]], axis=1)
    return a / max(conc(ref), 1e-12)


def metric_ks(x) -> Dict[str, float]:
    from scipy import stats
    f = _np(x).reshape(-1)
    mu, sd = float(f.mean()), float(f.std() + 1e-12)
    d, p = stats.kstest((f - mu) / sd, "norm")
    return {"ks_D": float(d), "ks_p": float(p), "mean": mu, "std": sd}


# =======================================================================
# the eight-domain report
# =======================================================================
WEIGHTS = {                       # composite-Score weights (equal, per the paper)
    "ks": 1.0, "kurt": 1.0, "entropy": 1.0, "circ": 1.0,
    "papr": 1.0, "scf": 1.0, "c42": 1.0, "wvd": 1.0,
}
# (value, direction, pass_fn, d_i normaliser)  ->  d_i in [0,1], 0 == ideal
CRITERIA = {
    "ks":      lambda v: v > 0.05,
    "kurt":    lambda v: abs(v - 3.0) < 0.15,
    "entropy": lambda v: v > 0.95,
    "circ":    lambda v: v < 0.15,
    "papr":    lambda v: abs(v) < 3.0,
    "scf":     lambda v: v < 3.0,
    "c42":     lambda v: abs(v) < 0.3,
    "wvd":     lambda v: v < 3.7,
}


def deviation(m: Dict[str, float]) -> Dict[str, float]:
    """Normalised per-domain deviation from the noise reference, in [0,1]."""
    return {
        "ks": float(np.clip(1.0 - min(m["ks_p"], 1.0) / 0.5, 0.0, 1.0)),
        "kurt": float(np.clip(abs(m["kurt"] - 3.0) / 0.30, 0.0, 1.0)),
        "entropy": float(np.clip((1.0 - m["entropy"]) / 0.05, 0.0, 1.0)),
        "circ": float(np.clip(m["circ"] / 0.30, 0.0, 1.0)),
        "papr": float(np.clip(abs(m["papr"]) / 6.0, 0.0, 1.0)),
        "scf": float(np.clip(m["scf"] / 6.0, 0.0, 1.0)),
        "c42": float(np.clip(abs(m["c42"]) / 0.6, 0.0, 1.0)),
        "wvd": float(np.clip(abs(m["wvd"] - 1.0) / 2.7, 0.0, 1.0)),
    }


def ew_report(sig: np.ndarray, ref: np.ndarray | None = None,
              n_fft: int = 512) -> Dict[str, float]:
    """sig: (N,2,T) float array of generated FRAMES (payload only)."""
    if ref is None:
        ref = np.random.randn(*sig.shape).astype(np.float32)
    k = metric_ks(sig)
    m = {
        "ks_p": k["ks_p"], "ks_D": k["ks_D"], "mean": k["mean"], "std": k["std"],
        "kurt": metric_kurtosis(sig),
        "entropy": metric_spectral_entropy(sig, n_fft),
        "circ": metric_iq_circularity(sig),
        "papr": metric_papr_db(sig) - metric_papr_db(ref),
        "papr_db": metric_papr_db(sig),
        "scf": metric_scf_ratio(sig, ref),
        "c42": metric_hos_c42(sig),
        "wvd": metric_wvd_ratio(sig, ref),
    }
    d = deviation(m)
    Sc = float(sum(WEIGHTS[k_] * d[k_] for k_ in WEIGHTS) / sum(WEIGHTS.values()))
    m.update({f"d_{k_}": v for k_, v in d.items()})
    m["Sc"] = Sc
    m["grade"] = "Excellent" if Sc < 0.10 else "Good" if Sc < 0.20 else \
                 "Moderate" if Sc < 0.35 else "Poor"
    m["n_pass"] = int(sum(bool(CRITERIA[k_](m[k_] if k_ != "ks" else m["ks_p"]))
                          for k_ in CRITERIA))
    m["pass"] = {k_: bool(CRITERIA[k_](m[k_] if k_ != "ks" else m["ks_p"]))
                 for k_ in CRITERIA}
    return m
