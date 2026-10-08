"""Synchronization module for LPI waveform detection.
Includes Zadoff-Chu CAZAC pilot generation, matched-filter frame timing synchronization,
coarse CFO estimation, and decision-directed phase correction.
"""

from typing import Tuple
import numpy as np


def generate_zadoff_chu(length: int = 512, root: int = 25) -> np.ndarray:
    """Generates a constant amplitude zero autocorrelation (CAZAC) Zadoff-Chu sequence."""
    n = np.arange(length)
    if length % 2 == 0:
        zc = np.exp(-1j * np.pi * root * (n ** 2) / length)
    else:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    # Unit average power
    zc = zc / np.sqrt(np.mean(np.abs(zc)**2))
    return zc.astype(np.complex64)


class TimingSynchronizer:
    """Detects frame boundaries using matched-filter correlation with Zadoff-Chu pilot."""

    def __init__(self, pilot: np.ndarray = None, length: int = 512, root: int = 25):
        if pilot is None:
            self.pilot = generate_zadoff_chu(length, root)
        else:
            self.pilot = pilot.astype(np.complex64)
        self.pilot_len = len(self.pilot)
        # Matched filter impulse response is time-reversed complex conjugate
        self.mf_kernel = np.conj(self.pilot[::-1])

    def detect_peak(self, rx_signal: np.ndarray, threshold_ratio: float = 3.0) -> Tuple[int, float, np.ndarray]:
        """Correlates rx_signal with pilot and finds the peak sample index.
        Returns:
            (peak_index, peak_snr_db, correlation_magnitude)
        """
        if len(rx_signal) < self.pilot_len:
            return 0, 0.0, np.array([])

        corr = np.correlate(rx_signal, self.pilot, mode='valid')
        mag = np.abs(corr)
        peak_idx = int(np.argmax(mag))
        peak_val = mag[peak_idx]

        # Calculate noise floor excluding the peak neighborhood
        mask = np.ones(len(mag), dtype=bool)
        guard = max(16, self.pilot_len // 16)
        start_guard = max(0, peak_idx - guard)
        end_guard = min(len(mag), peak_idx + guard)
        mask[start_guard:end_guard] = False

        if np.any(mask):
            noise_floor = np.median(mag[mask]) + 1e-12
        else:
            noise_floor = np.mean(mag) + 1e-12

        peak_snr = 20.0 * np.log10(peak_val / noise_floor)
        return peak_idx, float(peak_snr), mag


class CFOEstimator:
    """Estimates and compensates Carrier Frequency Offset (CFO) and phase rotation."""

    def __init__(self, fs: float = 2e6):
        self.fs = fs

    def estimate_cfo_from_pilot(self, rx_pilot: np.ndarray, ref_pilot: np.ndarray) -> float:
        """Estimates CFO (Hz) by computing angle progression over halves of the pilot sequence."""
        L = min(len(rx_pilot), len(ref_pilot))
        half = L // 2
        # Cross-correlate first half with second half
        p1 = rx_pilot[:half] * np.conj(ref_pilot[:half])
        p2 = rx_pilot[half:2*half] * np.conj(ref_pilot[half:2*half])

        # Phase difference over T = half / fs
        r = np.sum(p2 * np.conj(p1))
        phase_diff = np.angle(r)
        cfo_hz = phase_diff / (2.0 * np.pi * (half / self.fs))
        return float(cfo_hz)

    def compensate_cfo(self, signal: np.ndarray, cfo_hz: float, initial_phase: float = 0.0) -> np.ndarray:
        """Removes CFO and initial phase rotation from signal."""
        t = np.arange(len(signal)) / self.fs
        rot = np.exp(-1j * (2.0 * np.pi * cfo_hz * t + initial_phase))
        return (signal * rot).astype(np.complex64)

    def estimate_residual_phase(self, rx_pilot: np.ndarray, ref_pilot: np.ndarray) -> float:
        """Estimates constant residual phase angle from matched pilot."""
        L = min(len(rx_pilot), len(ref_pilot))
        dot = np.sum(rx_pilot[:L] * np.conj(ref_pilot[:L]))
        return float(np.angle(dot))
