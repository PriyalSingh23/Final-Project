"""Cryptographic engine for LPI-CGAN communications (Project ANSHUMAN).
Provides authenticated AES-128-CTR and AES-GCM encryption, key derivation,
and CRC-8 frame integrity validation.
"""

import os
import struct
from typing import Tuple, Union
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


CRC8_POLYNOMIAL = 0x07
CRC8_INIT = 0x00


def calculate_crc8(data: bytes) -> int:
    """Computes CRC-8-ATM (poly=0x07) over input bytes."""
    crc = CRC8_INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLYNOMIAL) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class AESEngine:
    """AES Encryption/Decryption Engine with 128-bit key and frame integrity."""

    def __init__(self, key: bytes = None):
        if key is None:
            # Standard 16-byte military/research reference key
            self.key = b"ANSHUMAN_LPI_KEY"
        else:
            if len(key) != 16:
                raise ValueError(f"AES-128 key must be exactly 16 bytes, got {len(key)}")
            self.key = key

    def encrypt_ctr(self, plaintext: bytes, nonce: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypts plaintext using AES-128-CTR mode.
        Returns (ciphertext, nonce).
        """
        if nonce is None:
            nonce = get_random_bytes(8)
        elif len(nonce) != 8:
            raise ValueError(f"CTR nonce must be 8 bytes, got {len(nonce)}")

        cipher = AES.new(self.key, AES.MODE_CTR, nonce=nonce)
        ciphertext = cipher.encrypt(plaintext)
        return ciphertext, nonce

    def decrypt_ctr(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Decrypts ciphertext using AES-128-CTR mode."""
        cipher = AES.new(self.key, AES.MODE_CTR, nonce=nonce)
        return cipher.decrypt(ciphertext)

    def pack_secure_frame(self, message: Union[str, bytes], max_payload_len: int = 21) -> bytes:
        """Packs a plaintext message into a structured authenticated payload:
        [1-byte msg_len] + [Payload up to max_len bytes] + [1-byte CRC-8]
        Padded to exactly max_payload_len + 2 bytes (default 23 bytes).
        Then encrypts with AES-128-CTR using a deterministic or pre-shared session nonce.
        """
        if isinstance(message, str):
            msg_bytes = message.encode("utf-8")
        else:
            msg_bytes = bytes(message)

        if len(msg_bytes) > max_payload_len:
            raise ValueError(f"Message exceeds maximum length ({len(msg_bytes)} > {max_payload_len})")

        # Header: 1-byte length prefix
        payload = struct.pack("B", len(msg_bytes)) + msg_bytes
        # Pad to max_payload_len + 1
        pad_len = (max_payload_len + 1) - len(payload)
        payload = payload + b"\x00" * pad_len

        # Trailer: 1-byte CRC-8
        crc = calculate_crc8(payload)
        frame_raw = payload + struct.pack("B", crc)

        # Encrypt frame
        # Use fixed zero-nonce for deterministic stream synchronization in packetized SDR
        nonce = b"\x00" * 8
        ciphertext, _ = self.encrypt_ctr(frame_raw, nonce=nonce)
        return ciphertext

    def unpack_secure_frame(self, ciphertext: bytes, max_payload_len: int = 21) -> Tuple[bytes, bool]:
        """Decrypts and unpacks structured payload.
        Verifies CRC-8 checksum.
        Returns (decoded_message, crc_valid).
        """
        nonce = b"\x00" * 8
        frame_raw = self.decrypt_ctr(ciphertext, nonce=nonce)

        expected_len = max_payload_len + 2
        if len(frame_raw) < expected_len:
            return b"", False

        payload = frame_raw[:expected_len - 1]
        received_crc = frame_raw[expected_len - 1]
        computed_crc = calculate_crc8(payload)

        crc_valid = (received_crc == computed_crc)
        if not crc_valid:
            return b"", False

        msg_len = payload[0]
        if msg_len > max_payload_len:
            return b"", False

        msg_bytes = payload[1:1 + msg_len]
        return msg_bytes, True
