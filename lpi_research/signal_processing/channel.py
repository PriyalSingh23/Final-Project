"""Calibrated 5-Impairment RF Channel Simulator.
Simulates real-world over-the-air channel effects:
  1. AWGN (calibrated SNR in dB)
  2. Carrier Frequency Offset (CFO in Hz)
  3. Wiener Phase Noise (rad/sample)
  4. DC Offset (I/Q leakage)
  5. IQ Imbalance (amplitude mismatch and phase skew)
  6. Sample Arrival Delay (timing jitter)
"""

import numpy as np


class RFChannelSimulator:
    """Configurable RF channel impairment generator."""

    def __init__(self,
                 fs: float = 2e6,
                 cfo_hz: float = 0.0,
                 snr_db: float = None,
                 phase_noise_std: float = 0.0,
                 dc_offset: complex = 0.0 + 0.0j,
                 iq_amp_imbalance_db: float = 0.0,
                 iq_phase_imbalance_deg: float = 0.0,
                 sample_delay: int = 0):
        self.fs = fs
        self.cfo_hz = cfo_hz
        self.snr_db = snr_db
        self.phase_noise_std = phase_noise_std
        self.dc_offset = dc_offset
        self.iq_amp_imbalance_db = iq_amp_imbalance_db
        self.iq_phase_imbalance_deg = iq_phase_imbalance_deg
        self.sample_delay = sample_delay

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Applies configured RF impairments to the input baseband complex signal."""
        out = signal.astype(np.complex64).copy()
        N = len(out)

        # 1. Carrier Frequency Offset (CFO)
        if abs(self.cfo_hz) > 1e-9:
            t = np.arange(N) / self.fs
            out = out * np.exp(1j * 2.0 * np.pi * self.cfo_hz * t)

        # 2. Wiener Phase Noise
        if self.phase_noise_std > 0:
            d_phi = np.random.normal(0, self.phase_noise_std, N)
            phi = np.cumsum(d_phi)
            out = out * np.exp(1j * phi)

        # 3. IQ Imbalance (Amplitude & Phase)
        if abs(self.iq_amp_imbalance_db) > 1e-6 or abs(self.iq_phase_imbalance_deg) > 1e-6:
            g = 10.0 ** (self.iq_amp_imbalance_db / 20.0)
            phi_skew = np.deg2rad(self.iq_phase_imbalance_deg)
            # Model: I' = I * (1 + g/2), Q' = Q * (1 - g/2) + phase skew
            i_comp = out.real * np.cos(phi_skew / 2.0) - out.imag * np.sin(phi_skew / 2.0)
            q_comp = -out.real * np.sin(phi_skew / 2.0) + out.imag * np.cos(phi_skew / 2.0)
            out = (g * i_comp + 1j * (1.0 / g) * q_comp).astype(np.complex64)

        # 4. DC Offset
        if abs(self.dc_offset) > 1e-9:
            out = out + self.dc_offset

        # 5. AWGN
        if self.snr_db is not None:
            sig_power = np.mean(np.abs(out)**2)
            snr_linear = 10.0 ** (self.snr_db / 10.0)
            noise_power = sig_power / (snr_linear + 1e-12)
            noise_sigma = np.sqrt(noise_power / 2.0)
            noise = (np.random.normal(0, noise_sigma, N) +
                     1j * np.random.normal(0, noise_sigma, N))
            out = out + noise

        # 6. Sample Arrival Delay (pad leading/trailing with noise)
        if self.sample_delay > 0:
            prefix_noise = (np.random.normal(0, 0.707, self.sample_delay) +
                            1j * np.random.normal(0, 0.707, self.sample_delay))
            postfix_noise = (np.random.normal(0, 0.707, 500) +
                             1j * np.random.normal(0, 0.707, 500))
            out = np.concatenate([prefix_noise, out, postfix_noise])

        return out.astype(np.complex64)
