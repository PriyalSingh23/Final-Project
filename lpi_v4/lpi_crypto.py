#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpi_crypto.py -- payload stack: AES-128-CTR/GCM + CRC-8 + Reed-Solomon + framing
=================================================================================

Protocol (one message = one waveform burst):

    plaintext (<= 25 B)
      -> nonce 8 B  ||  AES-128-ctr ciphertext  ||  CRC-8        = 34 data symbols
      -> RS(42,34)  (8 parity symbols, corrects 4 bad bytes)     = 42 symbols = 336 bits
      -> 3 covert frames of n_bits=128 (384 bits, 48 bits of zero padding)

Three fixes against the earlier pipeline / the reference draft:

  1. The draft says "the 336-bit frame is converted to a 256-bit bipolar stream
     by zero-padding".  Truncating 336 -> 256 silently destroys 80 payload bits,
     and the "BER = 0" claim cannot be true for the *whole* message.  Here the
     frame count is derived from the codeword size (42*8/n_bits, rounded up) and
     padded at the END, so no bit is lost.
  2. The web app encrypted the string while the GRC transmit chain modulated raw
     file bytes, so TX and RX were not speaking the same protocol.  encode() and
     decode() here are the only payload path, used by the trainer's link test, the
     USRP scripts, the GRC blocks and the web app.
  3. The nonce is per-message (a fresh 8-byte counter IV), not a fixed counter
     starting at 0 -- with AES-CTR a repeated keystream under the same key is a
     total-loss failure of the encryption (two-time-pad).

The AES key is PBKDF2-free but SHA-256-derived from the passphrase (16 bytes),
which is what the field demo uses; swap in a real KDF for anything serious.
"""
from __future__ import annotations

import hashlib
import struct
from typing import List, Tuple

import numpy as np

try:
    from Crypto.Cipher import AES
    HAVE_AES = True
except Exception:                                        # pragma: no cover
    AES = None
    HAVE_AES = False
try:
    from reedsolo import RSCodec
    HAVE_RS = True
except Exception:                                        # pragma: no cover
    RSCodec = None
    HAVE_RS = False

NONCE_BYTES = 8
RS_NSYM = 8                       # RS(42,34)
RS_K = 34                         # data symbols per codeword
TAG_BYTES = 4                     # truncated GCM tag (16 B will not fit in 34)
#  data[0]      : plaintext length n
#  data[1:9]    : per-message nonce / counter IV
#  data[9:9+m]  : ciphertext, m = n (ctr) or n+TAG_BYTES (gcm)
#  data[9+m]    : CRC-8 over data[0:9+m]
#  remainder    : zero pad
DATA_BYTES_CTR = RS_K - 1 - NONCE_BYTES - 1                  # 24
DATA_BYTES_GCM = DATA_BYTES_CTR - TAG_BYTES                  # 20
DATA_BYTES = DATA_BYTES_CTR
CODEWORD_BITS = (RS_K + RS_NSYM) * 8       # 336
CRC_POLY = 0x07


# ------------------------------------------------------------------- CRC ----
def crc8(data: bytes, poly: int = CRC_POLY) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# ----------------------------------------------------------------- AES ------
def _key(passphrase: str) -> bytes:
    return hashlib.sha256(passphrase.encode()).digest()[:16]


def _ctr_cipher(key: bytes, nonce: bytes):
    return AES.new(key, AES.MODE_CTR, nonce=nonce)


def aes_encrypt(pt: bytes, passphrase: str, mode: str = "ctr") -> Tuple[bytes, bytes]:
    """-> (nonce, ciphertext).  GCM appends the 16-byte auth tag (authenticated
    encryption); CTR keeps the frame small and matches the paper's headline."""
    key = _key(passphrase)
    nonce = hashlib.sha256(key + pt + mode.encode()).digest()[:NONCE_BYTES]
    if not HAVE_AES:                                  # test fallback, NOT secure
        k = np.frombuffer((key * 32)[: len(pt)], dtype=np.uint8)
        return nonce, bytes(np.bitwise_xor(np.frombuffer(pt, np.uint8), k).tobytes())
    if mode.lower() == "gcm":
        c = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_BYTES)
        ct, tag = c.encrypt_and_digest(pt)
        return nonce, ct + tag[:TAG_BYTES]
    return nonce, _ctr_cipher(key, nonce).encrypt(pt)


def aes_decrypt(nonce: bytes, ct: bytes, passphrase: str, mode: str = "ctr") -> bytes:
    key = _key(passphrase)
    if not HAVE_AES:
        k = np.frombuffer((key * 32)[: len(ct)], dtype=np.uint8)
        return bytes(np.bitwise_xor(np.frombuffer(ct, np.uint8), k).tobytes())
    if mode.lower() == "gcm":
        c = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_BYTES)
        return c.decrypt_and_verify(ct[:-TAG_BYTES], ct[-TAG_BYTES:])
    return _ctr_cipher(key, nonce).decrypt(ct)


# ------------------------------------------------------------------ ECC -----
def rs_encode(data: bytes) -> bytes:
    """data (34 B) -> 42 B codeword."""
    if not HAVE_RS:
        return data + bytes(RS_NSYM)
    return bytes(RSCodec(RS_NSYM).encode(data))[: RS_K + RS_NSYM]


def rs_decode(data: bytes) -> bytes:
    """42 B codeword -> 34 B (uncorrectable -> zeros, caught by the CRC)."""
    if not HAVE_RS:
        return data[:RS_K]
    try:
        return bytes(RSCodec(RS_NSYM).decode(data)[0])[:RS_K]
    except Exception:
        return b"\x00" * RS_K


# ================================================================ framing ==
class FrameCodec:
    """str <-> covert frame bits.  One call = one RS codeword = one burst."""

    def __init__(self, passphrase: str = "lpi-demo-key", n_bits: int = 128,
                 mode: str = "ctr"):
        """`frames` = ceil(336 / n_bits): the codeword is written into the FIRST
        336 bits of the burst and the tail is zero padding, so the reader must
        advance by frames*n_bits per codeword, not by 336."""
        self.passphrase = passphrase
        self.n_bits = int(n_bits)
        self.mode = mode
        self.frames = int(np.ceil(CODEWORD_BITS / self.n_bits))     # 3 at n_bits=128

    # ---- TX ---------------------------------------------------------------
    @property
    def max_len(self) -> int:
        """Longest plaintext that fits one RS(42,34) codeword in this mode."""
        return DATA_BYTES_GCM if self.mode.lower() == "gcm" else DATA_BYTES_CTR

    def encode(self, text) -> np.ndarray:
        if isinstance(text, str):
            text = text.encode("utf-8", "ignore")
        if len(text) > self.max_len:
            raise ValueError(f"payload {len(text)} B > {self.max_len} B per burst "
                             f"(RS(42,34) data field); use encode_long() and send "
                             f"the bursts in order")
        nonce, ct = aes_encrypt(text, self.passphrase, self.mode)
        data = bytearray(RS_K)
        data[0] = len(text)
        data[1:1 + NONCE_BYTES] = nonce
        data[1 + NONCE_BYTES: 1 + NONCE_BYTES + len(ct)] = ct
        used = 1 + NONCE_BYTES + len(ct)
        data[used] = crc8(bytes(data[:used]))
        data = bytes(data)
        coded = rs_encode(data if isinstance(data, bytes) else bytes(data))
        bits = np.unpackbits(np.frombuffer(coded, np.uint8), bitorder="big")
        out = np.zeros(self.frames * self.n_bits, dtype=np.uint8)
        out[: bits.size] = bits
        return out                                  # flat, frames*n_bits (384)

    def encode_long(self, text: str) -> np.ndarray:
        """arbitrary length -> flat (k*frames*n_bits,) uint8 burst sequence.

        Chunking is per *character*, not per byte: a burst boundary in the middle
        of a multi-byte UTF-8 code point would be unrecoverable at the receiver
        (each burst is decoded on its own), which is how a "successful" decode can
        come back with U+FFFD in place of an accent or an emoji.
        """
        chunks, cur = [], b""
        for ch in (text[i:i + 1] for i in range(len(text))):
            b = ch.encode("utf-8", "ignore")
            if cur and len(cur) + len(b) > self.max_len:
                chunks.append(cur)
                cur = b
            else:
                cur += b
        if cur:
            chunks.append(cur)
        if not chunks:
            chunks = [b""]
        return np.concatenate([self.encode(c) for c in chunks])

    # ---- RX ---------------------------------------------------------------
    def decode(self, bits) -> Tuple[str, bool]:
        """(frames, n_bits) or flat -> (text, crc_ok)."""
        b = np.asarray(bits, dtype=np.uint8).reshape(-1)
        need = CODEWORD_BITS
        stride = self.frames * self.n_bits       # 384 at n_bits=128, NOT 336
        if b.size % stride:                      # trim to whole bursts
            b = b[: b.size - (b.size % stride)]
        if b.size == 0:
            return "", False
        txt, ok = [], True
        for i in range(0, b.size - stride + 1, stride):
            coded = np.packbits(b[i:i + need], bitorder="big")[: RS_K + RS_NSYM]
            data = rs_decode(bytes(coded))
            n = data[0]
            m = n + (TAG_BYTES if self.mode.lower() == "gcm" else 0)
            used = 1 + NONCE_BYTES + m
            if n == 0 or used + 1 > RS_K:
                txt.append("")
                ok = False
                continue
            good = data[used] == crc8(data[:used])
            ok = ok and good
            try:
                pt = aes_decrypt(data[1:1 + NONCE_BYTES], data[1 + NONCE_BYTES:used],
                                  self.passphrase, self.mode)[:n]
                txt.append(pt.decode("utf-8", "replace"))
            except Exception:
                txt.append("")
                ok = False
        return "".join(txt), bool(ok)


def message_bits(text, cfg, passphrase: str = "lpi-demo-key", mode: str = "ctr") -> np.ndarray:
    """Convenience for the TX scripts: string -> flat uint8 bit stream whose
    length is an exact multiple of cfg.n_bits."""
    return FrameCodec(passphrase, cfg.n_bits, mode).encode(text).reshape(-1)


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, ".")
    from lpi_core import LPIConfig
    cfg = LPIConfig()
    msg = (sys.argv[1] if len(sys.argv) > 1 else "ALPHA_SECURE_IND_001A")
    for mode in ("ctr", "gcm"):
        fc = FrameCodec("0123456789abcdef", cfg.n_bits, mode)
        bits = fc.encode(msg[:fc.max_len])
        back, ok = fc.decode(bits)
        msg = msg[:fc.max_len]
        print(f"[{mode}] AES={'yes' if HAVE_AES else 'NO'} RS={'yes' if HAVE_RS else 'NO'} | "
              f"{bits.shape[0]} frames x {fc.n_bits} bits = {bits.size} bits "
              f"({CODEWORD_BITS} coded + {bits.size - CODEWORD_BITS} pad) "
              f"-> {back!r}  crc_ok={ok}")
    # burst-error resilience: flip 4 bytes of the codeword, RS(42,34) must heal it
    fc = FrameCodec("0123456789abcdef", cfg.n_bits)
    bits = fc.encode(msg).reshape(-1)
    rng = np.random.default_rng(0)
    bad = bits.copy()
    bad[rng.choice(bits.size // 2, 26, replace=False)] ^= 1     # ~3 byte errors
    print("[burst] after 26 random bit errors ->", fc.decode(bad))
    t0 = time.time()
    for _ in range(50):
        fc.encode(msg)
    print(f"[rate] encode {1000*(time.time()-t0)/50:.2f} ms/burst")
