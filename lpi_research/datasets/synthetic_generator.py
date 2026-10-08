"""Synthetic Multi-Modulation Dataset Generator matching RadioML & EW benchmarks.
Generates controlled IQ datasets with metadata:
  - sample ID
  - modulation
  - sample rate
  - center frequency
  - SNR
  - channel model
  - CFO
  - timing offset
  - phase offset
"""

import os
from typing import Dict, Any, Tuple, List
import numpy as np
import h5py

MODULATIONS = ['AWGN', 'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'CPFSK', 'OFDM', 'DSSS']


def generate_single_sample(mod: str,
                           n_samples: int = 512,
                           snr_db: float = 10.0,
                           cfo_hz: float = 0.0,
                           phase_deg: float = 0.0,
                           fs: float = 2e6) -> np.ndarray:
    """Synthesizes one baseband IQ sample array of length n_samples."""
    if mod == 'AWGN':
        sig = (np.random.normal(0, 0.707, n_samples) +
               1j * np.random.normal(0, 0.707, n_samples)).astype(np.complex64)
    elif mod == 'BPSK':
        syms = np.random.choice([-1.0, 1.0], size=n_samples) + 0j
        sig = syms.astype(np.complex64)
    elif mod == 'QPSK':
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)
        sig = np.random.choice(const, size=n_samples).astype(np.complex64)
    elif mod == '8PSK':
        angles = np.arange(8) * (2 * np.pi / 8)
        const = np.exp(1j * angles)
        sig = np.random.choice(const, size=n_samples).astype(np.complex64)
    elif mod == '16QAM':
        grid = np.array([-3, -1, 1, 3])
        i_comp = np.random.choice(grid, size=n_samples)
        q_comp = np.random.choice(grid, size=n_samples)
        sig = ((i_comp + 1j * q_comp) / np.sqrt(10.0)).astype(np.complex64)
    elif mod == '64QAM':
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i_comp = np.random.choice(grid, size=n_samples)
        q_comp = np.random.choice(grid, size=n_samples)
        sig = ((i_comp + 1j * q_comp) / np.sqrt(42.0)).astype(np.complex64)
    elif mod == 'CPFSK':
        bits = np.random.choice([-1, 1], size=n_samples)
        phase = np.cumsum(bits * (np.pi / 4))
        sig = np.exp(1j * phase).astype(np.complex64)
    elif mod == 'OFDM':
        n_sub = 64
        n_blocks = int(np.ceil(n_samples / n_sub))
        blocks = []
        for _ in range(n_blocks):
            qpsk = (np.random.choice([-1, 1], n_sub) + 1j * np.random.choice([-1, 1], n_sub)) / np.sqrt(2.0)
            t_block = np.fft.ifft(qpsk) * np.sqrt(n_sub)
            blocks.append(t_block)
        sig = np.concatenate(blocks)[:n_samples].astype(np.complex64)
    elif mod == 'DSSS':
        sf = 32
        n_bits = int(np.ceil(n_samples / sf))
        bits = np.random.choice([-1, 1], n_bits)
        pn = np.random.choice([-1, 1], sf)
        sig = np.kron(bits, pn)[:n_samples].astype(np.complex64)
    else:
        sig = (np.random.normal(0, 0.707, n_samples) +
               1j * np.random.normal(0, 0.707, n_samples)).astype(np.complex64)

    # Normalize power to 1.0
    sig = sig / (np.sqrt(np.mean(np.abs(sig)**2)) + 1e-12)

    # Apply CFO and initial phase
    t = np.arange(n_samples) / fs
    rot = np.exp(1j * (2.0 * np.pi * cfo_hz * t + np.deg2rad(phase_deg)))
    sig = (sig * rot).astype(np.complex64)

    # Apply AWGN noise based on SNR
    if mod != 'AWGN':
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        noise = (np.random.normal(0, noise_sigma, n_samples) +
                 1j * np.random.normal(0, noise_sigma, n_samples))
        sig = (sig + noise).astype(np.complex64)

    return sig


def generate_dataset_hdf5(filepath: str,
                          samples_per_mod: int = 200,
                          n_samples: int = 512,
                          snr_range: List[float] = [-10, 0, 10, 20]) -> str:
    """Creates an HDF5 dataset file matching RadioML / Project ANSHUMAN format."""
    total_samples = len(MODULATIONS) * samples_per_mod * len(snr_range)
    print(f"[Dataset] Generating {total_samples} samples into {filepath}...")

    # Data arrays: (N, 2, n_samples)
    x_data = np.zeros((total_samples, 2, n_samples), dtype=np.float32)
    y_labels = []
    snr_vals = np.zeros(total_samples, dtype=np.float32)

    idx = 0
    for mod_idx, mod in enumerate(MODULATIONS):
        for snr in snr_range:
            for _ in range(samples_per_mod):
                cfo = float(np.random.uniform(-50.0, 50.0))
                phase = float(np.random.uniform(0.0, 360.0))
                sig = generate_single_sample(mod, n_samples=n_samples, snr_db=snr, cfo_hz=cfo, phase_deg=phase)

                x_data[idx, 0, :] = sig.real
                x_data[idx, 1, :] = sig.imag
                y_labels.append(mod)
                snr_vals[idx] = snr
                idx += 1

    with h5py.File(filepath, "w") as f:
        f.create_dataset("X", data=x_data, compression="gzip")
        # Store labels as fixed-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("labels", data=np.array(y_labels, dtype=object), dtype=dt)
        f.create_dataset("SNR", data=snr_vals)
        f.attrs["num_samples"] = total_samples
        f.attrs["frame_len"] = n_samples
        f.attrs["modulations"] = MODULATIONS

    print(f"[Dataset] Successfully saved HDF5 to {filepath} ({os.path.getsize(filepath) / 1e6:.2f} MB)")
    return filepath
