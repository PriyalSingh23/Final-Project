"""Unit tests for cryptographic engine and CRC-8 integrity.
"""

import pytest
from lpi_research.crypto.aes_engine import AESEngine, calculate_crc8


def test_crc8_calculation():
    data = b"ANSHUMAN_LPI_TEST"
    crc = calculate_crc8(data)
    assert isinstance(crc, int)
    assert 0 <= crc <= 255
    # Determinism
    assert calculate_crc8(data) == crc
    # Bit sensitivity
    corrupted = bytearray(data)
    corrupted[0] ^= 0x01
    assert calculate_crc8(corrupted) != crc


def test_aes_ctr_encryption_decryption():
    engine = AESEngine()
    msg = b"SECRET_RADAR_DATA_12345"
    ciphertext, nonce = engine.encrypt_ctr(msg)
    assert ciphertext != msg
    assert len(nonce) == 8
    decrypted = engine.decrypt_ctr(ciphertext, nonce)
    assert decrypted == msg


def test_pack_unpack_secure_frame():
    engine = AESEngine()
    plaintext = "ALPHA_SECURE_01"
    cipher_frame = engine.pack_secure_frame(plaintext, max_payload_len=21)
    # Total frame size: max_payload_len (21) + len_byte (1) + crc_byte (1) = 23 bytes
    assert len(cipher_frame) == 23

    recovered_bytes, crc_valid = engine.unpack_secure_frame(cipher_frame, max_payload_len=21)
    assert crc_valid is True
    assert recovered_bytes.decode("utf-8") == plaintext


def test_corrupted_frame_detection():
    engine = AESEngine()
    plaintext = "ALPHA_SECURE_01"
    cipher_frame = bytearray(engine.pack_secure_frame(plaintext, max_payload_len=21))
    # Flip a bit
    cipher_frame[5] ^= 0xFF
    recovered_bytes, crc_valid = engine.unpack_secure_frame(bytes(cipher_frame), max_payload_len=21)
    assert crc_valid is False
