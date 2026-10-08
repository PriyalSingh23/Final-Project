#!/usr/bin/env python3
"""
make_bits_file.py -- write the *coded* payload that tx_lpi_v4.grc transmits.
============================================================================

The GRC transmitter takes one byte stream and turns every byte into 8 BPSK
symbols, so the crypto + error-correction has to happen *before* the flowgraph:
this tool applies the same FrameCodec the USRP scripts use (CRC-8 -> RS(42,34)
-> AES-128-CTR) and writes the resulting codeword bytes, zero-padded so the
frame count is a multiple of frames-per-burst.

    python make_bits_file.py --text "ALPHA-INDIA-001" --out /tmp/lpi_bits.dat
    python make_bits_file.py --file msg.txt --bursts 20 --out /tmp/lpi_bits.dat
    python make_bits_file.py --print-bits --text hi

Then set the TX flowgraph's `tx_bits_file` variable to that path (or point a
`blocks_file_source` at it) and the two chains agree bit for bit -- the
consistency test for that is `python lpi_grc.py` / `tests/test_link.py`.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpi_crypto import FrameCodec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="LPI-v4 SHADOWCOMM GRC TEST 001")
    ap.add_argument("--file", default="", help="read the message from a text file")
    ap.add_argument("--out", default="/tmp/lpi_bits.dat")
    ap.add_argument("--key", default="SESSION-KEY")
    ap.add_argument("--mode", default="ctr", choices=["ctr", "gcm"])
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--bursts", type=int, default=1, help="repeat the burst N times")
    ap.add_argument("--print-bits", action="store_true")
    a = ap.parse_args()

    msg = open(a.file).read() if a.file else a.text
    fc = FrameCodec(a.key, a.n_bits, mode=a.mode)
    bits = fc.encode_long(msg)                       # uint8, one value per bit
    per_burst = fc.frames * a.n_bits
    n_bursts = max(1, a.bursts)
    bits = np.tile(bits, n_bursts * (per_burst // max(1, bits.size)) or 1)
    if bits.size % per_burst:                        # pad to whole bursts
        bits = np.concatenate([bits, np.zeros(per_burst - bits.size % per_burst, np.uint8)])
    # MSB-first packing, i.e. exactly blocks.unpack_k_bits_bb(k=8) in reverse
    data = np.packbits(bits, bitorder="big")
    if a.print_bits:
        print("".join(str(int(b)) for b in bits[:128]), "...")
    with open(a.out, "wb") as f:
        f.write(data.tobytes())
    print(f"[bits] {a.out}: {data.size} bytes = {data.size * 8} bits = "
          f"{data.size * 8 // a.n_bits} frames ({a.n_bits} bits each) | "
          f"msg {len(msg)} chars -> {bits.size // per_burst} burst(s), "
          f"key={a.key!r} mode={a.mode}")
    print(f"[bits] on-air: {data.size * 8 // a.n_bits} frames x 576 samples = "
          f"{data.size * 8 // a.n_bits * 576 / 245760 * 1e3:.1f} ms at 245760 S/s")


if __name__ == "__main__":
    raise SystemExit(main())
