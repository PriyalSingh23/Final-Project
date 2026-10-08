"""Baseline reference waveforms (BPSK, QPSK, DSSS, AWGN) for EW benchmark comparison.
"""

import numpy as np


def generate_awgn(num_samples: int, power: float = 1.0) -> np.ndarray:
    """Generates pure complex AWGN with total power = power (sigma^2 / 2 per component)."""
    sigma = np.sqrt(power / 2.0)
    noise = np.random.normal(0, sigma, num_samples) + 1j * np.random.normal(0, sigma, num_samples)
    return noise.astype(np.complex64)


def generate_bpsk(num_samples: int, sps: int = 4, roll_off: float = 0.35) -> np.ndarray:
    """Generates BPSK waveform with RRC filtering."""
    num_symbols = int(np.ceil(num_samples / sps)) + 16
    bits = np.random.randint(0, 2, num_symbols)
    symbols = (2 * bits - 1).astype(np.complex64)

    # Upsample
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    # Simple root-raised-cosine or gaussian pulse shaping
    t = np.arange(-3 * sps, 3 * sps + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * roll_off * t) / (1 - (2 * roll_off * t)**2 + 1e-12)
    h = h / np.sqrt(np.sum(h**2))

    filtered = np.convolve(upsampled, h, mode='same')
    out = filtered[:num_samples]
    # Normalize power to 1.0
    out = out / (np.sqrt(np.mean(np.abs(out)**2)) + 1e-12)
    return out.astype(np.complex64)


def generate_qpsk(num_samples: int, sps: int = 4, roll_off: float = 0.35) -> np.ndarray:
    """Generates QPSK waveform with pulse shaping."""
    num_symbols = int(np.ceil(num_samples / sps)) + 16
    bits_i = np.random.randint(0, 2, num_symbols) * 2 - 1
    bits_q = np.random.randint(0, 2, num_symbols) * 2 - 1
    symbols = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2.0)

    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    t = np.arange(-3 * sps, 3 * sps + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * roll_off * t) / (1 - (2 * roll_off * t)**2 + 1e-12)
    h = h / np.sqrt(np.sum(h**2))

    filtered = np.convolve(upsampled, h, mode='same')
    out = filtered[:num_samples]
    out = out / (np.sqrt(np.mean(np.abs(out)**2)) + 1e-12)
    return out.astype(np.complex64)


def generate_dsss(num_samples: int, spreading_factor: int = 64) -> np.ndarray:
    """Generates Direct-Sequence Spread Spectrum (DSSS) using pseudo-random PN code."""
    num_chips = num_samples
    num_bits = int(np.ceil(num_chips / spreading_factor))
    bits = np.random.randint(0, 2, num_bits) * 2 - 1
    pn_sequence = np.random.randint(0, 2, spreading_factor) * 2 - 1

    spread_chips = np.zeros(num_bits * spreading_factor, dtype=np.complex64)
    for i in range(num_bits):
        spread_chips[i * spreading_factor:(i + 1) * spreading_factor] = bits[i] * pn_sequence

    out = spread_chips[:num_samples]
    out = out / (np.sqrt(np.mean(np.abs(out)**2)) + 1e-12)
    return out.astype(np.complex64)
