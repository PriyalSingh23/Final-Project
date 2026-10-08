"""Integration test verifying complete end-to-end communication pipeline:
TX -> RF Channel (AWGN + CFO + Timing Delay) -> RX -> 100% Bit-Perfect Recovery.
"""

from lpi_research.sdr.interface import LPIBurstTransmitter, LPIBurstReceiver
from lpi_research.signal_processing.channel import RFChannelSimulator


def test_end_to_end_transceiver_recovery():
    tx = LPIBurstTransmitter()
    rx = LPIBurstReceiver()

    secret_message = "ALPHA_SECURE_IND_001A"

    # 1. Transmitter
    burst_iq, tx_meta = tx.transmit_burst(secret_message)
    assert len(burst_iq) == 1024

    # 2. Channel with impairments: SNR=18 dB, CFO=+25 Hz, delay=135 samples
    sim_delay = 135
    channel = RFChannelSimulator(cfo_hz=25.0, snr_db=18.0, sample_delay=sim_delay)
    rx_signal = channel.apply(burst_iq)

    # 3. Receiver
    recovered_text, rx_telemetry = rx.receive_burst(rx_signal)

    # Assertions
    assert rx_telemetry["success"] is True
    assert rx_telemetry["crc_passed"] is True
    assert rx_telemetry["peak_index"] == sim_delay
    assert recovered_text == secret_message
