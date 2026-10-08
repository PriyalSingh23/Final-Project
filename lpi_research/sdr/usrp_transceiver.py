"""USRP Hardware Transceiver & Over-The-Air Verification for LPI-CGAN.
Executes physical RF transmission and reception using UHD (USRP B200 / B210 / B200mini).
Verifies:
  1. USRP Device Probe & RF Front-End Tuning
  2. Over-the-Air LPI Burst Transmission (AES-128 + RS(32,23) + CGAN Baseband + ZC Sync)
  3. Physical RF Reception, Matched-Filter Synchronization, & CFO Derotation
  4. Neural Decoding & Bit-Perfect Message Recovery
  5. 8-Domain Statistical Electronic Warfare (EW) Stealth Verification
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# Ensure repository root is in search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lpi_research.sdr.interface import LPIBurstTransmitter, LPIBurstReceiver
from lpi_research.gan.metrics import evaluate_lpi_signal
from lpi_research.waveform.baseline import generate_awgn

try:
    import uhd  # type: ignore
    UHD_AVAILABLE = True
except ImportError:
    UHD_AVAILABLE = False


def probe_usrp(args_str: str = "") -> Optional[Dict[str, Any]]:
    """Probes connected USRP hardware and returns front-end specifications."""
    if not UHD_AVAILABLE:
        print("[ERR] UHD Python package is not available.")
        return None

    try:
        devs = uhd.find(args_str)
        if not devs:
            return None

        dev_info = devs[0]
        usrp = uhd.usrp.MultiUSRP(args_str)
        
        info = {
            "type": dev_info.get("type", "Unknown USRP"),
            "serial": dev_info.get("serial", "Unknown Serial"),
            "name": dev_info.get("name", ""),
            "master_clock_rate": usrp.get_master_clock_rate(),
            "tx_rates": usrp.get_tx_rates(0),
            "rx_rates": usrp.get_rx_rates(0),
            "tx_antennas": usrp.get_tx_antennas(0),
            "rx_antennas": usrp.get_rx_antennas(0),
            "tx_gain_range": (usrp.get_tx_gain_range(0).start(), usrp.get_tx_gain_range(0).stop()),
            "rx_gain_range": (usrp.get_rx_gain_range(0).start(), usrp.get_rx_gain_range(0).stop()),
            "tx_freq_range": (usrp.get_tx_freq_range(0).start(), usrp.get_tx_freq_range(0).stop()),
            "rx_freq_range": (usrp.get_rx_freq_range(0).start(), usrp.get_rx_freq_range(0).stop()),
        }
        return info
    except Exception as e:
        print(f"[ERR] Error probing USRP: {e}")
        return None


def run_usrp_tx_rx_loopback(message: str = "USRP_LPI_SECURE_01",
                            freq_hz: float = 750e6,
                            rate_hz: float = 1e6,
                            tx_gain: float = 25.0,
                            rx_gain: float = 35.0,
                            args_str: str = "") -> bool:
    """Executes full RF transmit & receive check over physical USRP hardware."""
    print("=" * 75)
    print("PROJECT ANSHUMAN -- USRP PHYSICAL HARDWARE TX/RX VERIFICATION")
    print("=" * 75)

    if not UHD_AVAILABLE:
        print("[FAIL] UHD Python library is missing.")
        return False

    devs = uhd.find(args_str)
    if not devs:
        print("[FAIL] No USRP device found on USB or Network interface.")
        print("       Ensure the WinUSB driver is bound to the USRP and USB cable is connected.")
        return False

    print(f"[HW] USRP Detected: {devs[0].get('type', 'B200/B210')} (Serial: {devs[0].get('serial', 'N/A')})")

    # 1. Initialize MultiUSRP
    usrp = uhd.usrp.MultiUSRP(args_str)
    usrp.set_tx_rate(rate_hz, 0)
    usrp.set_rx_rate(rate_hz, 0)
    actual_rate = usrp.get_tx_rate(0)

    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq_hz), 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq_hz), 0)
    actual_freq = usrp.get_tx_freq(0)

    usrp.set_tx_gain(tx_gain, 0)
    usrp.set_rx_gain(rx_gain, 0)
    
    # Antennas
    tx_antennas = usrp.get_tx_antennas(0)
    rx_antennas = usrp.get_rx_antennas(0)
    if "TX/RX" in tx_antennas:
        usrp.set_tx_antenna("TX/RX", 0)
    if "RX2" in rx_antennas:
        usrp.set_rx_antenna("RX2", 0)
    elif "TX/RX" in rx_antennas:
        usrp.set_rx_antenna("TX/RX", 0)

    print(f"[RF] Center Frequency: {actual_freq / 1e6:.3f} MHz")
    print(f"[RF] Sample Rate:      {actual_rate / 1e6:.3f} Msps")
    print(f"[RF] TX Gain:          {tx_gain:.1f} dB  | TX Antenna: {usrp.get_tx_antenna(0)}")
    print(f"[RF] RX Gain:          {rx_gain:.1f} dB  | RX Antenna: {usrp.get_rx_antenna(0)}")
    print("-" * 75)

    # 2. Modulate Message into LPI-CGAN Burst
    tx = LPIBurstTransmitter()
    burst_iq, tx_meta = tx.transmit_burst(message)
    print(f"[TX] Message:          '{message}' ({len(message)} chars)")
    print(f"[TX] Modulated Burst:  {len(burst_iq)} complex samples (512 ZC Pilot + 512 Covert Payload)")

    # Pad burst with silence to allow transient settling
    pad_len = 1000
    tx_samples = np.pad(burst_iq, (pad_len, pad_len), mode="constant").astype(np.complex64)

    # 3. Setup Streamers
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)

    tx_meta_uhd = uhd.types.TXMetadata()
    tx_meta_uhd.has_time_spec = False
    tx_meta_uhd.start_of_burst = True
    tx_meta_uhd.end_of_burst = True

    rx_buffer = np.zeros(len(tx_samples) + 4096, dtype=np.complex64)
    rx_meta_uhd = uhd.types.RXMetadata()

    # 4. Transmit & Receive
    print("[HW] Streaming over-the-air burst through USRP...")
    rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    rx_stream_cmd.num_samps = len(rx_buffer)
    rx_stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(rx_stream_cmd)

    # Send TX burst
    tx_streamer.send(tx_samples, tx_meta_uhd)

    # Receive RX buffer
    samps_received = rx_streamer.recv(rx_buffer, rx_meta_uhd)
    print(f"[HW] Captured:         {samps_received} complex samples from RF front-end")

    # 5. Receiver Synchronization & Decoding
    rx = LPIBurstReceiver(fs=actual_rate)
    recovered, rx_meta = rx.receive_burst(rx_buffer[:samps_received])

    print("-" * 75)
    print(f"[RX] Detected Sync:    Sample {rx_meta['peak_index']} [SNR: {rx_meta['peak_snr_db']:.1f} dB]")
    print(f"[RX] Estimated CFO:    {rx_meta['cfo_hz']:+.2f} Hz")
    print(f"[RX] Phase Rotation:   {rx_meta['residual_phase_deg']:+.2f} deg")
    print(f"[RX] RS Errors Fixed:  {rx_meta['rs_corrected_bytes']} byte(s)")
    print(f"[RX] CRC-8 Check:      {'PASSED [OK]' if rx_meta['crc_passed'] else 'FAILED'}")
    print(f"[RX] Decrypted Text:   '{recovered}'")
    is_success = (recovered == message)
    print(f"[RX] Physical Verdict: {'SUCCESS (100% BIT-PERFECT RECOVERY)' if is_success else 'FAILED'}")

    # 6. Statistical EW Stealth Check on Radiated Payload
    payload = burst_iq[512:]
    ref_awgn = generate_awgn(len(payload))
    ew = evaluate_lpi_signal(payload, ref_awgn=ref_awgn)
    print("-" * 75)
    print(f"[EW] Kolmogorov-Smirnov p: {ew['ks_p']:.4f} (Ideal: 1.0, Target: > 0.05)")
    print(f"[EW] Kurtosis:             {ew['kurtosis']:.4f} (Target: ~ 3.00 Gaussian)")
    print(f"[EW] Spectral Entropy:     {ew['spectral_entropy']:.4f} (Target: > 0.95)")
    print(f"[EW] Stealth Grade:        {ew['stealth_grade']} (Score: {ew['composite_stealth_score']:.4f})")
    print("=" * 75)

    return is_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USRP Hardware Transceiver for LPI-CGAN")
    parser.add_argument("--probe", action="store_true", help="Probe connected USRP hardware only")
    parser.add_argument("--msg", default="USRP_LPI_SECURE_01", help="Message to transmit")
    parser.add_argument("--freq", type=float, default=750e6, help="Center frequency in Hz (default: 750 MHz)")
    parser.add_argument("--rate", type=float, default=1e6, help="Sample rate in Hz (default: 1.0 Msps)")
    parser.add_argument("--tx-gain", type=float, default=25.0, help="TX Gain in dB")
    parser.add_argument("--rx-gain", type=float, default=35.0, help="RX Gain in dB")
    parser.add_argument("--args", default="", help="UHD device address args (e.g. 'type=b200')")
    args = parser.parse_args()

    if args.probe:
        info = probe_usrp(args.args)
        if info:
            print("USRP Device Specifications:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print("No USRP device detected.")
    else:
        run_usrp_tx_rx_loopback(
            message=args.msg,
            freq_hz=args.freq,
            rate_hz=args.rate,
            tx_gain=args.tx_gain,
            rx_gain=args.rx_gain,
            args_str=args.args
        )
