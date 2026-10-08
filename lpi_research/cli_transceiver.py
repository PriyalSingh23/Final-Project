"""Command-line interface to execute LPI transmission, simulated RF channel,
and receiver synchronization/decoding with full 8-domain EW evaluation.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lpi_research.sdr.interface import LPIBurstTransmitter, LPIBurstReceiver
from lpi_research.signal_processing.channel import RFChannelSimulator
from lpi_research.gan.metrics import evaluate_lpi_signal
from lpi_research.waveform.baseline import generate_awgn


def main():
    parser = argparse.ArgumentParser(description="CLI LPI Burst Transceiver")
    parser.add_argument("--msg", default="ALPHA_SECURE_IND_001A", help="Plaintext message (max 21 bytes)")
    parser.add_argument("--snr", type=float, default=18.0, help="Channel SNR in dB")
    parser.add_argument("--cfo", type=float, default=25.0, help="CFO in Hz")
    parser.add_argument("--delay", type=int, default=150, help="Arrival delay in samples")
    parser.add_argument("--save-iq", default="tx_burst.bin", help="Output IQ file path")
    args = parser.parse_args()

    print("=" * 70)
    print("PROJECT ANSHUMAN -- LPI-CGAN SECURE WAVEFORM TRANSCEIVER")
    print(f"Plaintext:         '{args.msg}' ({len(args.msg)} bytes)")
    print(f"RF Channel Impair: SNR={args.snr:.1f} dB, CFO={args.cfo:.1f} Hz, Delay={args.delay} smp")
    print("-" * 70)

    # 1. Transmitter
    tx = LPIBurstTransmitter()
    burst_iq, tx_meta = tx.transmit_burst(args.msg)
    burst_iq.tofile(args.save_iq)
    print(f"[TX] Generated:     {len(burst_iq)} samples (512 pilot + 512 payload)")
    print(f"[TX] Saved baseband: {args.save_iq}")

    # 2. EW Stealth Evaluation
    payload_iq = burst_iq[512:]
    ref_awgn = generate_awgn(len(payload_iq))
    ew = evaluate_lpi_signal(payload_iq, ref_awgn=ref_awgn)
    print(f"[TX Stealth] KS p-val:    {ew['ks_p']:.4f} {'[PASS]' if ew['pass_ks'] else '[FAIL]'}")
    print(f"[TX Stealth] Kurtosis:    {ew['kurtosis']:.4f} {'[PASS]' if ew['pass_kurtosis'] else '[FAIL]'}")
    print(f"[TX Stealth] Entropy:     {ew['spectral_entropy']:.4f} {'[PASS]' if ew['pass_entropy'] else '[FAIL]'}")
    print(f"[TX Stealth] Circularity: {ew['iq_circularity']:.4f} {'[PASS]' if ew['pass_circularity'] else '[FAIL]'}")
    print(f"[TX Stealth] Score Sc:    {ew['composite_stealth_score']:.4f} [{ew['stealth_grade']}]")
    print("-" * 70)

    # 3. Channel Simulation
    channel = RFChannelSimulator(cfo_hz=args.cfo, snr_db=args.snr, sample_delay=args.delay)
    rx_signal = channel.apply(burst_iq)
    print(f"[Channel] Output:   {len(rx_signal)} samples captured")

    # 4. Receiver
    rx = LPIBurstReceiver()
    recovered, rx_meta = rx.receive_burst(rx_signal)
    print(f"[RX] Detected Sync: Sample {rx_meta['peak_index']} (True {args.delay}) [Peak SNR: {rx_meta['peak_snr_db']:.1f} dB]")
    print(f"[RX] Estimated CFO: {rx_meta['cfo_hz']:+.2f} Hz (True {args.cfo:+.2f} Hz)")
    print(f"[RX] RS Corrected:  {rx_meta['rs_corrected_bytes']} byte(s)")
    print(f"[RX] CRC-8 Check:   {'PASSED [OK]' if rx_meta['crc_passed'] else 'FAILED'}")
    print(f"[RX] Recovered:     '{recovered}'")
    match = (recovered == args.msg)
    print(f"[RX] Bit-Perfect:   {'SUCCESS (100% BIT-PERFECT)' if match else 'MISMATCH'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
