"""Unit tests for the conventional QPSK baseline communication chain.
"""

from lpi_research.waveform.qpsk_system import QPSKTransmitter, QPSKReceiver
from lpi_research.signal_processing.channel import RFChannelSimulator


def test_qpsk_baseline_clean_channel():
    tx = QPSKTransmitter()
    rx = QPSKReceiver()
    test_msg = "ALPHA_SECURE_01"

    burst_iq, _ = tx.modulate(test_msg)
    chan = RFChannelSimulator(snr_db=25.0, cfo_hz=0.0, sample_delay=0)
    rx_iq = chan.apply(burst_iq)

    rec_msg, tel = rx.demodulate(rx_iq)
    assert tel["success"] is True
    assert tel["crc_passed"] is True
    assert rec_msg == test_msg


def test_qpsk_baseline_with_rf_impairments():
    tx = QPSKTransmitter()
    rx = QPSKReceiver()
    test_msg = "ALPHA_SECURE_01"

    burst_iq, _ = tx.modulate(test_msg)
    # Impairments: CFO=20 Hz, Delay=100 samples, SNR=18 dB
    chan = RFChannelSimulator(snr_db=18.0, cfo_hz=20.0, sample_delay=100)
    rx_iq = chan.apply(burst_iq)

    rec_msg, tel = rx.demodulate(rx_iq)
    assert tel["success"] is True
    assert tel["crc_passed"] is True
    assert tel["peak_index"] == 100
    assert rec_msg == test_msg
