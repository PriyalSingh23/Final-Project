#!/usr/bin/env python3
"""
rx_decode_final_v2.py

Decodes a real captured signal from your B205mini-i loopback test back to
the original message: timing sync, PN despread, RRC matched filter, BPSK
slice, bit packing, Reed-Solomon decode, AES-CTR decode.

USAGE:
  1. Edit CAPTURE_FILE and ORIGINAL_MESSAGE (+ params if changed) below.
  2. Run: python rx_decode_final_v2.py
"""

import numpy as np
from gnuradio.filter import firdes
from gnuradio import digital, gr, blocks
from Crypto.Cipher import AES
from Crypto.Util import Counter
from reedsolo import RSCodec, ReedSolomonError

# ============================== CONFIG ==============================
CAPTURE_FILE = r"C:\Users\yasht\Desktop\GNU Radio\Blocks practice\raw_capture.bin"
ORIGINAL_MESSAGE = b"HELLO_WORLD_THIS_IS_A_TEST_MESSAGE_FOR_SDR_TRANSMISSION\n"
AES_KEY_HEX = "000102030405060708090A0B0C0D0E0F"
AES_NONCE_HEX = "00000000000000000000000000000000"
RS_K, RS_NSYM = 56, 16
PN_DEGREE, PN_MASK, PN_SEED = 7, 0, 1
SPS, ALPHA, NTAPS = 4, 0.35, 44
# ======================================================================


def pn_chips(n, dtype=np.float32):
    class PNGen(gr.top_block):
        def __init__(self, n):
            gr.top_block.__init__(self)
            self.src = digital.glfsr_source_f(PN_DEGREE, True, PN_MASK, PN_SEED)
            self.head = blocks.head(gr.sizeof_float, n)
            self.sink = blocks.vector_sink_f()
            self.connect(self.src, self.head, self.sink)
    tb = PNGen(n)
    tb.run()
    return np.array(tb.sink.data(), dtype=dtype)


def regenerate_tx_reference():
    """Rebuild the exact TX waveform using REAL GNU Radio blocks (proven
    correct against your actual hardware flowgraph earlier in this project),
    for use as a timing-sync reference."""
    key = bytes.fromhex(AES_KEY_HEX)
    nonce = bytes.fromhex(AES_NONCE_HEX)
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, 'big'))
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    ciphertext = cipher.encrypt(ORIGINAL_MESSAGE)

    rsc = RSCodec(RS_NSYM)
    rs_encoded = bytearray()
    for i in range(0, len(ciphertext), RS_K):
        block = ciphertext[i:i+RS_K]
        if len(block) == RS_K:
            rs_encoded.extend(rsc.encode(bytes(block)))
    rs_encoded = bytes(rs_encoded)

    taps = firdes.root_raised_cosine(1, SPS, 1.0, ALPHA, NTAPS)

    class TXRef(gr.top_block):
        def __init__(self, data):
            gr.top_block.__init__(self)
            self.src = blocks.vector_source_b(list(data))
            self.unpack = blocks.unpack_k_bits_bb(8)
            self.c2s = digital.chunks_to_symbols_bc((-1+0j, 1+0j), 1)
            self.rrc = grfilter.interp_fir_filter_ccf(SPS, taps)
            self.pn_src = digital.glfsr_source_b(PN_DEGREE, True, PN_MASK, PN_SEED)
            self.pn_c2s = digital.chunks_to_symbols_bc((-1+0j, 1+0j), 1)
            self.mult = blocks.multiply_vcc(1)
            self.sink = blocks.vector_sink_c()
            self.connect(self.src, self.unpack, self.c2s, self.rrc, (self.mult, 0))
            self.connect(self.pn_src, self.pn_c2s, (self.mult, 1))
            self.connect(self.mult, self.sink)

    from gnuradio import filter as grfilter
    tb = TXRef(rs_encoded)
    tb.run()
    tx_ref = np.array(tb.sink.data(), dtype=np.complex64)
    return tx_ref, rs_encoded


def main():
    print("Regenerating expected TX reference (real GNU Radio blocks)...")
    tx_reference, rs_encoded_expected = regenerate_tx_reference()
    print(f"  Reference: {len(tx_reference)} samples ({len(rs_encoded_expected)} RS-encoded bytes)")

    print(f"\nLoading captured file: {CAPTURE_FILE}")
    try:
        captured = np.fromfile(CAPTURE_FILE, dtype=np.complex64)
    except FileNotFoundError:
        print("  FILE NOT FOUND. Use Shift+right-click 'Copy as path' in File Explorer for the exact path.")
        return
    print(f"  Captured: {len(captured)} samples")

    print("\nFinding timing sync...")
    correlation = np.correlate(captured, tx_reference, mode='valid')
    if len(correlation) == 0:
        print("  Capture too short -- increase your Head block's sample count and recapture.")
        return
    peak_offset = int(np.argmax(np.abs(correlation)))
    peak_val = np.abs(correlation[peak_offset])
    mean_val = np.mean(np.abs(correlation))
    ratio = peak_val / mean_val if mean_val > 0 else 0
    print(f"  Peak at offset {peak_offset}, peak/mean ratio {ratio:.1f}x")
    if ratio < 5:
        print("  WARNING: weak correlation peak -- signal may be absent, or capture window too tight.")

    aligned = captured[peak_offset:peak_offset + len(tx_reference)]

    print("\nDespreading...")
    chips = pn_chips(len(aligned))
    despread = aligned * chips

    print("Matched filtering (RRC, numpy convolve) and slicing to symbols...")
    taps = np.array(firdes.root_raised_cosine(1, SPS, 1.0, ALPHA, NTAPS))
    filtered = np.convolve(despread, taps, mode='full')
    group_delay = (len(taps) - 1) // 2
    start = 2 * group_delay
    n_symbols_expected = len(despread) // SPS
    symbols = filtered[start::SPS][:n_symbols_expected]
    print(f"  Recovered {len(symbols)} symbols")

    print("Slicing to bits and packing to bytes...")
    bits = (symbols.real > 0).astype(np.uint8)
    n_bytes = len(bits) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        val = 0
        for b in bits[i*8:(i+1)*8]:
            val = (val << 1) | int(b)
        packed[i] = val
    rs_encoded_recovered = packed.tobytes()
    print(f"  Packed {len(rs_encoded_recovered)} bytes")

    print("\nReed-Solomon decoding...")
    rsc = RSCodec(RS_NSYM)
    n_codeword = RS_K + RS_NSYM
    decoded_blocks = bytearray()
    for i in range(0, len(rs_encoded_recovered), n_codeword):
        block = rs_encoded_recovered[i:i+n_codeword]
        if len(block) < n_codeword:
            break
        try:
            decoded, _, errata = rsc.decode(block)
            decoded_blocks.extend(decoded)
            if errata:
                print(f"  Block at byte {i}: corrected {len(errata)} error(s)")
        except ReedSolomonError as e:
            print(f"  Block at byte {i}: UNCORRECTABLE -- {e}")
            decoded_blocks.extend(block[:RS_K])

    print("\nAES-CTR decoding...")
    key = bytes.fromhex(AES_KEY_HEX)
    nonce = bytes.fromhex(AES_NONCE_HEX)
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, 'big'))
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    recovered_message = cipher.decrypt(bytes(decoded_blocks))

    print()
    print("=" * 60)
    print(f"Expected:  {ORIGINAL_MESSAGE}")
    print(f"Recovered: {recovered_message}")
    print(f"EXACT MATCH: {recovered_message == ORIGINAL_MESSAGE}")
    print("=" * 60)


if __name__ == "__main__":
    main()