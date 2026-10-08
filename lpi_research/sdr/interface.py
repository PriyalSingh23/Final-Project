"""Software-Defined Radio (SDR) Interface for LPI Communications.
Supports:
  1. SimulationBackend: In-memory software loopback with calibrated RF channel impairments.
  2. FileBackend: Binary IQ file exchange (.bin, .fc32) with GNU Radio.
  3. USRPBackend: Real-time over-the-air transmission via UHD / USRP B200/B210.
"""

import os
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch

from ..crypto.aes_engine import AESEngine
from ..fec.reed_solomon import ReedSolomonEngine
from ..signal_processing.sync import generate_zadoff_chu
from ..gan.checkpoint_loader import load_trained_models


PILOT_LEN = 512
FRAME_LEN = 512
MSG_BITS_PER_FRAME = 256


class LPIBurstTransmitter:
    """End-to-end LPI Transmitter Pipeline:
    Plaintext -> AES-128 Encryption & CRC-8 -> RS(32, 23) Encoding ->
    CGAN Baseband Generator (z in R^64, b in {-1, +1}^256) -> Zadoff-Chu Frame Preamble.
    """

    def __init__(self, key: bytes = None, device: str = "cpu"):
        self.device = device
        self.aes = AESEngine(key=key)
        self.fec = ReedSolomonEngine(nsym=9)
        self.generator, _ = load_trained_models(device=device)
        self.pilot = generate_zadoff_chu(length=PILOT_LEN, root=25)

    def transmit_burst(self, message: Union[str, bytes]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encodes and modulates message into an LPI burst waveform.
        Returns:
            (burst_iq, telemetry_dict)
        """
        # 1. AES Encrypt + CRC-8 integrity (23 bytes payload)
        encrypted_frame = self.aes.pack_secure_frame(message, max_payload_len=21)

        # 2. Reed-Solomon FEC: 23 bytes -> 32 bytes (256 bits)
        fec_encoded = self.fec.encode(encrypted_frame)

        # 3. Unpack to bits and convert to bipolar [-1.0, +1.0]
        bits = np.unpackbits(np.frombuffer(fec_encoded, dtype=np.uint8)).astype(np.float32)
        b_bipolar = torch.from_numpy(bits * 2.0 - 1.0).unsqueeze(0).to(self.device)

        # 4. CGAN Waveform Generation (512 complex samples)
        z = torch.randn(1, 64, device=self.device)
        with torch.no_grad():
            iq_out = self.generator(z, b_bipolar).cpu().numpy()[0]
        payload_iq = (iq_out[0, :] + 1j * iq_out[1, :]).astype(np.complex64)

        # 5. Form complete burst: Pilot + Covert Payload
        burst_iq = np.concatenate([self.pilot, payload_iq]).astype(np.complex64)

        telemetry = {
            "plaintext_length": len(message),
            "ciphertext_length": len(encrypted_frame),
            "fec_length": len(fec_encoded),
            "pilot_samples": len(self.pilot),
            "payload_samples": len(payload_iq),
            "total_burst_samples": len(burst_iq),
        }
        return burst_iq, telemetry


class LPIBurstReceiver:
    """End-to-end LPI Receiver Pipeline:
    Raw IQ -> Timing Synchronization (Zadoff-Chu MF) -> CFO Estimation & Correction ->
    CGAN CNN Decoder -> Reed-Solomon Error Correction ->
    AES-128 Decryption & CRC-8 Authentication.
    """

    def __init__(self, key: bytes = None, fs: float = 2e6, device: str = "cpu"):
        self.fs = fs
        self.device = device
        self.aes = AESEngine(key=key)
        self.fec = ReedSolomonEngine(nsym=9)
        _, self.decoder = load_trained_models(device=device)
        self.pilot = generate_zadoff_chu(length=PILOT_LEN, root=25)

    def receive_burst(self, rx_iq: np.ndarray) -> Tuple[Optional[str], Dict[str, Any]]:
        """Processes received IQ stream, detects preamble, decodes, and decrypts message."""
        telemetry = {
            "success": False,
            "peak_index": -1,
            "peak_snr_db": 0.0,
            "cfo_hz": 0.0,
            "residual_phase_deg": 0.0,
            "rs_corrected_bytes": 0,
            "crc_passed": False,
            "decrypted_message": "",
        }

        # 1. Matched Filter Timing Synchronization
        corr = np.correlate(rx_iq, self.pilot, mode='valid')
        if len(corr) == 0:
            return None, telemetry

        mag = np.abs(corr)
        peak_idx = int(np.argmax(mag))
        peak_val = mag[peak_idx]

        noise_floor = np.median(mag) + 1e-12
        peak_snr = 20.0 * np.log10(peak_val / noise_floor)
        telemetry["peak_index"] = peak_idx
        telemetry["peak_snr_db"] = float(peak_snr)

        if len(rx_iq) - peak_idx < PILOT_LEN + FRAME_LEN:
            return None, telemetry

        # 2. Extract Pilot & CFO Estimation
        rx_pilot = rx_iq[peak_idx:peak_idx + PILOT_LEN]
        half = PILOT_LEN // 2
        p1 = rx_pilot[:half] * np.conj(self.pilot[:half])
        p2 = rx_pilot[half:] * np.conj(self.pilot[half:])
        phase_diff = np.angle(np.sum(p2) * np.conj(np.sum(p1)))
        est_cfo = float(phase_diff * self.fs / (2.0 * np.pi * half))
        telemetry["cfo_hz"] = est_cfo

        # 3. Carrier Phase Rotation Correction
        est_phase = float(np.angle(corr[peak_idx]))
        telemetry["residual_phase_deg"] = float(np.rad2deg(est_phase))

        n_axis = np.arange(len(rx_iq), dtype=np.float64)
        cfo_correction = np.exp(-1j * (2.0 * np.pi * est_cfo * n_axis / self.fs + est_phase)).astype(np.complex64)
        corrected_signal = rx_iq * cfo_correction

        # 4. Extract and Normalize Payload Frame
        payload_start = peak_idx + PILOT_LEN
        frame_samples = corrected_signal[payload_start:payload_start + FRAME_LEN]

        v = np.stack([frame_samples.real, frame_samples.imag], axis=0).astype(np.float32)
        v = v - v.mean(axis=1, keepdims=True)
        s = v.std(axis=1, keepdims=True) + 1e-8
        v_norm = v / s

        # 5. CGAN CNN Decoder
        v_tensor = torch.from_numpy(v_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.decoder(v_tensor)
            bits_hat = (probs > 0.5).int().cpu().numpy()[0]

        # 6. Pack to Bytes and Reed-Solomon Decode
        fec_bytes = np.packbits(bits_hat).tobytes()
        decoded_bytes, err_count, rs_ok = self.fec.decode(fec_bytes)
        telemetry["rs_corrected_bytes"] = err_count

        if not rs_ok:
            return None, telemetry

        # 7. AES Decryption & CRC-8 Authentication
        msg_bytes, crc_valid = self.aes.unpack_secure_frame(decoded_bytes, max_payload_len=21)
        telemetry["crc_passed"] = crc_valid

        if crc_valid:
            try:
                decoded_text = msg_bytes.decode("utf-8")
                telemetry["decrypted_message"] = decoded_text
                telemetry["success"] = True
                return decoded_text, telemetry
            except Exception:
                return None, telemetry

        return None, telemetry
