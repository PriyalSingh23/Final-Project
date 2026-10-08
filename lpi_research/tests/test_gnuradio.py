"""Integration test for GNU Radio 3.10 TopBlock pipeline.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
import numpy as np

try:
    from gnuradio import gr, blocks  # type: ignore
    from lpi_research.sdr.gr_runner import execute_gnuradio_burst, is_usrp_available
    from lpi_research.sdr.interface import LPIBurstTransmitter, LPIBurstReceiver
    HAS_GNURADIO = True
except ImportError:
    HAS_GNURADIO = False


@pytest.mark.skipif(not HAS_GNURADIO, reason="GNU Radio 3.10 runtime not available")
def test_gnuradio_topblock_execution():
    """Verify that native GNU Radio 3.10 C++ top block runs and passes samples."""
    test_iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    captured, meta = execute_gnuradio_burst(test_iq, snr_db=25.0, cfo_hz=0.0, delay_samples=0)
    assert len(captured) >= len(test_iq)
    assert "GNU Radio" in meta["engine"]
    assert meta["elapsed_ms"] >= 0.0


@pytest.mark.skipif(not HAS_GNURADIO, reason="GNU Radio 3.10 runtime not available")
def test_gnuradio_end_to_end_recovery():
    """Verify end-to-end transmission, GNU Radio C++ execution, sync, and bit-perfect recovery."""
    tx = LPIBurstTransmitter()
    rx = LPIBurstReceiver()

    msg = "UNIT_TEST_GRC_01"
    burst_iq, tx_meta = tx.transmit_burst(msg)

    # Pass through GNU Radio C++ scheduler with RF channel impairments
    captured_iq, meta = execute_gnuradio_burst(
        burst_iq, snr_db=20.0, cfo_hz=20.0, delay_samples=100
    )

    # Receiver decoding
    recovered, rx_meta = rx.receive_burst(captured_iq)

    assert rx_meta["success"] is True
    assert rx_meta["crc_passed"] is True
    assert recovered == msg
    assert abs(rx_meta["peak_index"] - 100) <= 2
