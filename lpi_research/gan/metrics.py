"""8-Domain Electronic Warfare (EW) Statistical Evaluation Suite.
Based on Project ANSHUMAN / Ambuj Sharma et al. (IEEE GCON 2026).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
from scipy.signal import spectrogram


def calculate_ks_test(signal: np.ndarray) -> Tuple[float, float]:
    """1. Kolmogorov-Smirnov test against standard normal distribution N(0, 1)."""
    flat = np.concatenate([signal.real.reshape(-1), signal.imag.reshape(-1)])
    norm_flat = (flat - np.mean(flat)) / (np.std(flat) + 1e-12)
    stat, p_val = stats.kstest(norm_flat, stats.norm.cdf)
    return float(stat), float(p_val)


def calculate_kurtosis(signal: np.ndarray) -> float:
    """2. Kurtosis of real and imaginary baseband components (Fisher=False -> 3.0 is Gaussian)."""
    flat = np.concatenate([signal.real.reshape(-1), signal.imag.reshape(-1)])
    norm_flat = (flat - np.mean(flat)) / (np.std(flat) + 1e-12)
    kurt = stats.kurtosis(norm_flat, fisher=False)
    return float(kurt)


def calculate_spectral_entropy(signal: np.ndarray) -> float:
    """3. Normalized Spectral Entropy H in [0, 1]. Flat AWGN spectrum -> 1.0."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    psds = np.abs(np.fft.fft(signal, axis=1))**2
    psd_m = np.mean(psds, axis=0)
    p_norm = psd_m / (np.sum(psd_m) + 1e-12)
    h = -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm))
    return float(h)


def calculate_iq_circularity(signal: np.ndarray) -> float:
    """4. IQ Circularity coefficient eta = |E[s^2]| / E[|s|^2]. Proper circular AWGN -> 0.0."""
    s = signal.reshape(-1)
    s_centered = s - np.mean(s)
    pseudo_cov = np.abs(np.mean(s_centered**2))
    var = np.mean(np.abs(s_centered)**2) + 1e-12
    eta = pseudo_cov / var
    return float(eta)


def calculate_papr(signal: np.ndarray) -> float:
    """5. Peak-to-Average Power Ratio (PAPR) in dB."""
    p_inst = np.abs(signal.reshape(-1))**2
    p_peak = np.max(p_inst)
    p_avg = np.mean(p_inst) + 1e-12
    return float(10.0 * np.log10(p_peak / p_avg))


def calculate_csfa_scf_ratio(signal: np.ndarray, num_lags: int = 32) -> float:
    """6. Cyclostationary Spectral Correlation Function (SCF) max-to-mean ratio."""
    s = signal.reshape(-1)
    N = len(s)
    scf_max = 0.0
    scf_sum = 0.0
    count = 0
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        if lag < N:
            c = s[lag:] * np.conj(s[:-lag])
            fft_c = np.abs(np.fft.fft(c))
            m = np.max(fft_c)
            avg = np.mean(fft_c) + 1e-12
            ratio = m / avg
            if ratio > scf_max:
                scf_max = ratio
            scf_sum += ratio
            count += 1
    return float(scf_max)


def calculate_hos_c42(signal: np.ndarray) -> float:
    """7. Higher-Order Statistics Cumulant C42 = cum4(s, s*, s, s*). AWGN -> 0.0."""
    s = signal.reshape(-1)
    s = (s - np.mean(s)) / (np.std(s) + 1e-12)
    m42 = np.mean(np.abs(s)**4)
    m20 = np.mean(s**2)
    m21 = np.mean(np.abs(s)**2)
    c42 = m42 - 2 * (m21**2) - np.abs(m20)**2
    return float(np.real(c42))


def calculate_wvd_ratio(signal: np.ndarray, fs: float = 2e6) -> float:
    """8. Time-Frequency energy concentration ratio via Spectrogram."""
    if signal.ndim == 1:
        s = signal
    else:
        s = signal[0]
    _, _, pxx = spectrogram(s, fs=fs, nperseg=64, noverlap=32, return_onesided=False)
    max_energy = np.max(pxx)
    mean_energy = np.mean(pxx) + 1e-12
    return float(max_energy / mean_energy)


def calculate_composite_stealth_score(metrics: Dict[str, float]) -> Tuple[float, str]:
    """Computes Composite Stealth Score Sc based on weights:
      - KS deviation: 20%
      - Kurtosis deviation: 20%
      - Spectral Entropy deficit: 15%
      - IQ Circularity: 10%
      - PAPR deviation: 10%
      - CSFA SCF penalty: 10%
      - HOS |C42|: 10%
      - WVD TF penalty: 5%
    """
    ks_p = metrics.get("ks_p", 1.0)
    kurt = metrics.get("kurtosis", 3.0)
    entropy = metrics.get("spectral_entropy", 1.0)
    circularity = metrics.get("iq_circularity", 0.0)
    papr_dev = metrics.get("papr_dev_db", 0.0)
    csfa_ratio = metrics.get("csfa_ratio", 1.0)
    c42 = metrics.get("c42", 0.0)
    wvd_ratio = metrics.get("wvd_ratio", 1.0)

    # Normalized penalties in [0, 1]
    p_ks = max(0.0, 1.0 - ks_p)
    p_kurt = min(1.0, abs(kurt - 3.0) / 1.5)
    p_ent = max(0.0, 1.0 - entropy)
    p_circ = min(1.0, circularity / 0.5)
    p_papr = min(1.0, abs(papr_dev) / 5.0)
    p_csfa = min(1.0, max(0.0, (csfa_ratio - 1.0) / 3.0))
    p_c42 = min(1.0, abs(c42) / 0.5)
    p_wvd = min(1.0, max(0.0, (wvd_ratio - 1.0) / 3.0))

    sc = (0.20 * p_ks +
          0.20 * p_kurt +
          0.15 * p_ent +
          0.10 * p_circ +
          0.10 * p_papr +
          0.10 * p_csfa +
          0.10 * p_c42 +
          0.05 * p_wvd)

    if sc < 0.10:
        grade = "Excellent"
    elif sc < 0.20:
        grade = "Good"
    elif sc < 0.35:
        grade = "Moderate"
    else:
        grade = "Unacceptable"

    return float(sc), grade


def evaluate_lpi_signal(signal: np.ndarray, ref_awgn: np.ndarray = None) -> Dict[str, Any]:
    """Runs the complete 8-domain EW evaluation suite on the input signal."""
    _, ks_p = calculate_ks_test(signal)
    kurt = calculate_kurtosis(signal)
    entropy = calculate_spectral_entropy(signal)
    circularity = calculate_iq_circularity(signal)
    papr = calculate_papr(signal)

    # Reference AWGN PAPR ~ 9.0 dB
    awgn_papr = 9.0
    if ref_awgn is not None:
        awgn_papr = calculate_papr(ref_awgn)
    papr_dev = papr - awgn_papr

    csfa = calculate_csfa_scf_ratio(signal)
    c42 = calculate_hos_c42(signal)
    wvd = calculate_wvd_ratio(signal)

    metrics = {
        "ks_p": ks_p,
        "kurtosis": kurt,
        "spectral_entropy": entropy,
        "iq_circularity": circularity,
        "papr_db": papr,
        "papr_dev_db": papr_dev,
        "csfa_ratio": csfa,
        "c42": c42,
        "wvd_ratio": wvd
    }

    sc, grade = calculate_composite_stealth_score(metrics)
    metrics["composite_stealth_score"] = sc
    metrics["stealth_grade"] = grade

    # Pass/Fail boolean tests
    metrics["pass_ks"] = ks_p > 0.05
    metrics["pass_kurtosis"] = abs(kurt - 3.0) < 0.25
    metrics["pass_entropy"] = entropy > 0.95
    metrics["pass_circularity"] = circularity < 0.15
    metrics["pass_papr"] = abs(papr_dev) < 3.0
    metrics["pass_csfa"] = csfa < 3.0
    metrics["pass_c42"] = abs(c42) < 0.30
    metrics["pass_wvd"] = wvd < 3.70
    metrics["domains_passed"] = sum([
        metrics["pass_ks"], metrics["pass_kurtosis"], metrics["pass_entropy"],
        metrics["pass_circularity"], metrics["pass_papr"], metrics["pass_csfa"],
        metrics["pass_c42"], metrics["pass_wvd"]
    ])

    return metrics
