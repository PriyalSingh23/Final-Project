"""Conventional Baseline Communication System (QPSK with CAZAC Synchronization).
Implements the reference communication chain:
Plaintext -> AES-128 -> CRC-8 -> Reed-Solomon -> Interleaver -> QPSK Modulation
-> RF Impairment Channel -> Timing Sync -> CFO Compensation -> QPSK Demodulation
-> Deinterleaver -> RS Decoder -> AES Decrypt -> Recovered Message.
"""

from typing import Tuple, Dict, Any, List, Union
import numpy as np

from ..crypto.aes_engine import AESEngine
from ..fec.reed_solomon import ReedSolomonEngine
from ..fec.interleaver import MatrixInterleaver
from ..signal_processing.sync import generate_zadoff_chu, TimingSynchronizer, CFOEstimator
from ..signal_processing.channel import RFChannelSimulator


class QPSKTransmitter:
    """Baseline QPSK Transmitter."""

    def __init__(self, key: bytes = None, sps: int = 4):
        self.sps = sps
        self.aes = AESEngine(key=key)
        self.fec = ReedSolomonEngine(nsym=9)
        self.interleaver = MatrixInterleaver(rows=16, cols=16)
        self.pilot = generate_zadoff_chu(length=512, root=25)

    def modulate(self, message: Union[str, bytes]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 1. AES Encrypt + CRC-8 integrity (23 bytes payload)
        encrypted_frame = self.aes.pack_secure_frame(message, max_payload_len=21)

        # 2. Reed-Solomon FEC: 23 bytes -> 32 bytes (256 bits)
        fec_encoded = self.fec.encode(encrypted_frame)

        # 3. Interleaving
        interleaved_bytes = self.interleaver.interleave_bytes(fec_encoded)
        bits = np.unpackbits(np.frombuffer(interleaved_bytes, dtype=np.uint8)).astype(np.float32)

        # 4. QPSK Mapping: 2 bits per symbol -> 128 symbols
        bits_i = bits[0::2] * 2.0 - 1.0
        bits_q = bits[1::2] * 2.0 - 1.0
        symbols = (bits_i + 1j * bits_q).astype(np.complex64) / np.sqrt(2.0)

        # 5. Pulse shaping: repeat each symbol sps times (rectangular pulse, zero ISI)
        payload_iq = np.repeat(symbols, self.sps).astype(np.complex64)

        # 6. Prepend Zadoff-Chu pilot (512 samples)
        burst_iq = np.concatenate([self.pilot, payload_iq]).astype(np.complex64)

        telemetry = {
            "bits": len(bits),
            "symbols": len(symbols),
            "payload_samples": len(payload_iq),
            "total_samples": len(burst_iq),
        }
        return burst_iq, telemetry


class QPSKReceiver:
    """Baseline QPSK Receiver."""

    def __init__(self, key: bytes = None, fs: float = 2e6, sps: int = 4):
        self.fs = fs
        self.sps = sps
        self.aes = AESEngine(key=key)
        self.fec = ReedSolomonEngine(nsym=9)
        self.interleaver = MatrixInterleaver(rows=16, cols=16)
        self.pilot = generate_zadoff_chu(length=512, root=25)
        self.sync = TimingSynchronizer(pilot=self.pilot)
        self.cfo_est = CFOEstimator(fs=fs)

    def demodulate(self, rx_iq: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        telemetry = {
            "success": False,
            "peak_index": -1,
            "cfo_hz": 0.0,
            "raw_ber": 1.0,
            "rs_corrected_bytes": 0,
            "crc_passed": False,
            "recovered_text": "",
        }

        # 1. Timing Synchronization
        corr = np.correlate(rx_iq, self.pilot, mode='valid')
        if len(corr) == 0:
            return "", telemetry

        peak_idx = int(np.argmax(np.abs(corr)))
        telemetry["peak_index"] = peak_idx

        if len(rx_iq) - peak_idx < 512 + 512:
            return "", telemetry

        # 2. CFO & Phase Compensation
        rx_pilot = rx_iq[peak_idx:peak_idx + 512]
        half = 256
        p1 = rx_pilot[:half] * np.conj(self.pilot[:half])
        p2 = rx_pilot[half:] * np.conj(self.pilot[half:])
        est_cfo = float(np.angle(np.sum(p2) * np.conj(np.sum(p1))) * self.fs / (2.0 * np.pi * half))
        telemetry["cfo_hz"] = est_cfo

        est_phase = float(np.angle(corr[peak_idx]))
        n_axis = np.arange(len(rx_iq), dtype=np.float64)
        cfo_correction = np.exp(-1j * (2.0 * np.pi * est_cfo * n_axis / self.fs + est_phase)).astype(np.complex64)
        corrected = rx_iq * cfo_correction

        # 3. Payload Demodulation via integrate-and-dump
        payload_start = peak_idx + 512
        payload_samples = corrected[payload_start:payload_start + 512]

        symbols_hat = np.mean(payload_samples.reshape(-1, self.sps), axis=1)

        # Hard QPSK decision
        bits_i = (symbols_hat.real > 0).astype(np.uint8)
        bits_q = (symbols_hat.imag > 0).astype(np.uint8)

        # Interleave I and Q bits back to serial stream
        recovered_bits = np.empty(256, dtype=np.uint8)
        recovered_bits[0::2] = bits_i
        recovered_bits[1::2] = bits_q

        # 4. Deinterleave
        deinterleaved_bits = self.interleaver.deinterleave_bits(recovered_bits)
        fec_bytes = np.packbits(deinterleaved_bits).tobytes()

        # 5. Reed-Solomon Decode
        decoded_bytes, err_count, rs_ok = self.fec.decode(fec_bytes)
        telemetry["rs_corrected_bytes"] = err_count
        if not rs_ok:
            return "", telemetry

        # 6. AES Decrypt & CRC-8
        msg_bytes, crc_valid = self.aes.unpack_secure_frame(decoded_bytes, max_payload_len=21)
        telemetry["crc_passed"] = crc_valid

        if crc_valid:
            try:
                recovered_text = msg_bytes.decode("utf-8")
                telemetry["recovered_text"] = recovered_text
                telemetry["success"] = True
                return recovered_text, telemetry
            except Exception:
                return "", telemetry

        return "", telemetry


def run_qpsk_monte_carlo(snrs: List[float] = [0, 5, 10, 15, 20], num_trials: int = 100) -> Dict[str, Any]:
    """Runs Monte Carlo simulations of the QPSK baseline system over varying SNRs."""
    tx = QPSKTransmitter()
    rx = QPSKReceiver()
    test_msg = "ALPHA_SECURE_01"

    results = {}
    for snr in snrs:
        successes = 0
        total_bit_errors = 0
        total_bits = 0

        for _ in range(num_trials):
            sig, _ = tx.modulate(test_msg)
            chan = RFChannelSimulator(snr_db=snr, cfo_hz=15.0, sample_delay=50)
            rx_sig = chan.apply(sig)

            rec_text, tel = rx.demodulate(rx_sig)
            if tel["success"] and rec_text == test_msg:
                successes += 1

        results[snr] = {
            "success_rate": successes / num_trials,
            "snr_db": snr,
            "trials": num_trials
        }
    return results
