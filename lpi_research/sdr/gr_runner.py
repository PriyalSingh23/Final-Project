"""GNU Radio 3.10 Standalone TopBlock Transceiver Runner for LPI-CGAN.
Executes transmission and reception using the native GNU Radio runtime scheduler.
Supports in-memory vector streaming, binary file logging, and UHD USRP hardware detection.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
from typing import Tuple, Dict, Any, Optional
import numpy as np

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from gnuradio import gr, blocks  # type: ignore
except ImportError:
    class _DummyTopBlock:
        def __init__(self, *args, **kwargs): pass
        def run(self): pass
        def connect(self, *args): pass
    class _DummyGR:
        top_block = _DummyTopBlock
        sizeof_gr_complex = 8
    gr = _DummyGR()  # type: ignore
    blocks = None  # type: ignore

from lpi_research.sdr.interface import LPIBurstTransmitter, LPIBurstReceiver
from lpi_research.signal_processing.channel import RFChannelSimulator


def is_usrp_available() -> bool:
    """Checks if any USRP hardware (B200, B210, N210, etc.) is connected."""
    try:
        import uhd  # type: ignore
        devs = uhd.find("")
        return len(devs) > 0
    except Exception:
        return False


class GNU_Radio_Loopback_TopBlock(gr.top_block):
    """GNU Radio 3.10 TopBlock linking custom LPI stream blocks in loopback."""

    def __init__(self, tx_iq_data: np.ndarray, output_file: Optional[str] = "gnuradio_received.bin"):
        super().__init__("LPI_CGAN_GNU_Radio_Pipeline", catch_exceptions=True)

        # 1. Source: Vector source feeding complex baseband
        self.src = blocks.vector_source_c(tx_iq_data.tolist(), repeat=False)

        # 2. In-memory Sink: Vector Sink collecting complex baseband
        self.vector_sink = blocks.vector_sink_c()
        self.connect(self.src, self.vector_sink)

        # 3. File Sink for persistent RF capture
        self.output_file = output_file
        if output_file:
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, output_file, False)
            self.file_sink.set_unbuffered(True)
            self.connect(self.src, self.file_sink)

    def get_captured_samples(self) -> np.ndarray:
        return np.array(self.vector_sink.data(), dtype=np.complex64)


def execute_gnuradio_burst(burst_iq: np.ndarray,
                           snr_db: float = 18.0,
                           cfo_hz: float = 25.0,
                           delay_samples: int = 150,
                           output_file: Optional[str] = "gnuradio_received.bin"
                           ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs input burst IQ through simulated RF channel and native GNU Radio C++ TopBlock."""
    # 1. Apply calibrated RF Channel Impairments
    channel = RFChannelSimulator(cfo_hz=cfo_hz, snr_db=snr_db, sample_delay=delay_samples)
    impaired_iq = channel.apply(burst_iq)

    # 2. Instantiate and run native GNU Radio TopBlock
    t0 = time.perf_counter()
    tb = GNU_Radio_Loopback_TopBlock(impaired_iq, output_file=output_file)
    tb.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Extract output samples
    captured_iq = tb.get_captured_samples()

    telemetry = {
        "engine": "GNU Radio 3.10 Native C++ Runtime",
        "elapsed_ms": float(np.round(elapsed_ms, 2)),
        "samples_passed": len(captured_iq),
        "output_file": output_file if output_file and os.path.exists(output_file) else None,
        "usrp_detected": is_usrp_available()
    }
    return captured_iq, telemetry


def run_gnuradio_pipeline(message: str = "ALPHA_SECURE_IND_001A",
                          snr_db: float = 18.0,
                          cfo_hz: float = 25.0,
                          delay_smp: int = 150):
    print("=" * 70)
    print("PROJECT ANSHUMAN -- GNU RADIO 3.10 LPI TRANSCEIVER PIPELINE")
    print(f"Plaintext:         '{message}'")
    print(f"Channel Sim:       SNR={snr_db:.1f} dB, CFO={cfo_hz:.1f} Hz, Delay={delay_smp} smp")
    print(f"Hardware Status:   USRP Connected = {is_usrp_available()}")
    print("-" * 70)

    # 1. Transmitter
    tx = LPIBurstTransmitter()
    burst_iq, tx_meta = tx.transmit_burst(message)
    print(f"[TX] Modulated:    {len(burst_iq)} samples (512 pilot + 512 payload)")

    # 2. Execute GNU Radio 3.10 Flowgraph
    out_bin = "gnuradio_received.bin"
    captured_iq, gr_telemetry = execute_gnuradio_burst(
        burst_iq, snr_db=snr_db, cfo_hz=cfo_hz, delay_samples=delay_smp, output_file=out_bin
    )
    print(f"[GNU Radio] Flowgraph finished in {gr_telemetry['elapsed_ms']:.2f} ms ({gr_telemetry['samples_passed']} samples)")

    # 3. Receiver: Sync, CFO de-rotation, Neural Decoder, RS ECC, AES decrypt
    rx = LPIBurstReceiver()
    recovered, rx_meta = rx.receive_burst(captured_iq)

    print("-" * 70)
    print(f"[RX] Detected Sync: Sample {rx_meta['peak_index']} (True {delay_smp}) [SNR: {rx_meta['peak_snr_db']:.1f} dB]")
    print(f"[RX] Estimated CFO: {rx_meta['cfo_hz']:+.2f} Hz (True {cfo_hz:+.2f} Hz)")
    print(f"[RX] RS Corrected:  {rx_meta['rs_corrected_bytes']} byte(s)")
    print(f"[RX] CRC-8 Check:   {'PASSED [OK]' if rx_meta['crc_passed'] else 'FAILED'}")
    print(f"[RX] Recovered:     '{recovered}'")
    is_success = (recovered == message)
    print(f"[RX] Verdict:       {'SUCCESS (100% BIT-PERFECT RECOVERY)' if is_success else 'FAILED'}")
    print("=" * 70)
    return is_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNU Radio LPI Runner")
    parser.add_argument("--msg", default="ALPHA_SECURE_IND_001A", help="Plaintext message")
    parser.add_argument("--snr", type=float, default=18.0, help="SNR in dB")
    parser.add_argument("--cfo", type=float, default=25.0, help="CFO in Hz")
    parser.add_argument("--delay", type=int, default=150, help="Delay in samples")
    args = parser.parse_args()

    run_gnuradio_pipeline(args.msg, args.snr, args.cfo, args.delay)
