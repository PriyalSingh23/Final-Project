"""Unit tests for Reed-Solomon FEC and matrix interleaver.
"""

import numpy as np
from lpi_research.fec.reed_solomon import ReedSolomonEngine
from lpi_research.fec.interleaver import MatrixInterleaver


def test_reed_solomon_clean_roundtrip():
    rs = ReedSolomonEngine(nsym=9)
    payload = b"A" * 23
    encoded = rs.encode(payload)
    assert len(encoded) == 32

    decoded, err_count, success = rs.decode(encoded)
    assert success is True
    assert err_count == 0
    assert decoded == payload


def test_reed_solomon_error_correction():
    rs = ReedSolomonEngine(nsym=9)
    payload = b"SECRET_RADAR_MESSAGE_23"
    encoded = bytearray(rs.encode(payload))

    # Inject 4 symbol errors (maximum correctable with nsym=9 is floor(9/2)=4)
    encoded[2] ^= 0xAA
    encoded[7] ^= 0x55
    encoded[14] ^= 0xFF
    encoded[28] ^= 0x12

    decoded, err_count, success = rs.decode(bytes(encoded))
    assert success is True
    assert err_count == 4
    assert decoded == payload


def test_matrix_interleaver_roundtrip():
    interleaver = MatrixInterleaver(rows=16, cols=16)
    bits = np.random.randint(0, 2, 256)
    interleaved = interleaver.interleave_bits(bits)
    deinterleaved = interleaver.deinterleave_bits(interleaved)
    assert np.array_equal(bits, deinterleaved)
