"""Unit tests for Zadoff-Chu synchronization and CFO estimation.
"""

import numpy as np
from lpi_research.signal_processing.sync import (
    generate_zadoff_chu, TimingSynchronizer, CFOEstimator
)


def test_zadoff_chu_cazac_properties():
    zc = generate_zadoff_chu(length=512, root=25)
    assert len(zc) == 512
    # Constant amplitude: |zc[n]| should be ~1.0
    amplitudes = np.abs(zc)
    assert np.allclose(amplitudes, 1.0, atol=1e-5)


def test_timing_synchronizer_exact_peak():
    zc = generate_zadoff_chu(length=512, root=25)
    sync = TimingSynchronizer(pilot=zc)

    # Pad with 100 samples of noise before pilot
    noise_pre = (np.random.normal(0, 0.2, 100) + 1j * np.random.normal(0, 0.2, 100)).astype(np.complex64)
    noise_post = (np.random.normal(0, 0.2, 200) + 1j * np.random.normal(0, 0.2, 200)).astype(np.complex64)
    rx_stream = np.concatenate([noise_pre, zc, noise_post])

    peak_idx, snr_db, _ = sync.detect_peak(rx_stream)
    assert peak_idx == 100
    assert snr_db > 15.0


def test_cfo_estimation_and_compensation():
    fs = 2e6
    cfo_hz = 30.0
    zc = generate_zadoff_chu(length=512, root=25)

    # Apply CFO
    t = np.arange(len(zc)) / fs
    cfo_distorted = zc * np.exp(1j * 2.0 * np.pi * cfo_hz * t)

    cfo_est = CFOEstimator(fs=fs)
    est_cfo = cfo_est.estimate_cfo_from_pilot(cfo_distorted, zc)
    # Estimate should be within 5 Hz of ground truth
    assert abs(est_cfo - cfo_hz) < 5.0
