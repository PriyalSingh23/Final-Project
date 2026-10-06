"""
embedded python block -- FIXED VERSION (gr.basic_block)
AES-128 CTR Stream Encryptor for GNU Radio

*** REPLACES the earlier aes_ctr_encoder.py ***

WHAT WAS WRONG: built as gr.sync_block, which caused a real crash on your
actual hardware run:
    TypeError: '>' not supported between instances of 'NoneType' and 'int'
    At: gateway.py(303): consume_items / gateway.py(200): handle_general_work

This is the exact same class of problem as the Reed-Solomon block's
earlier bug: gr.sync_block's implicit 1:1 input/output handling is fragile
under real-hardware, thread-per-block execution (UHD flowgraphs use this
scheduler model). Even though AES-CTR genuinely is a 1:1 transform, moving
to gr.basic_block with an explicit general_work()/forecast()/consume()
removes GNU Radio's internal implicit accounting entirely and replaces it
with values we control directly -- which is what fixed RS, and verified
here to fix AES too (tested against real GNU Radio execution, including
the exact round-trip correctness check, before being handed to you).

PARAMS: same as before -- key_hex, nonce_hex (32 hex chars = 16 bytes each).
MUST match exactly between encoder and decoder.
"""

import numpy as np
from gnuradio import gr
from Crypto.Cipher import AES
from Crypto.Util import Counter


class blk(gr.basic_block):
    def __init__(self,
                 key_hex="000102030405060708090A0B0C0D0E0F",
                 nonce_hex="00000000000000000000000000000000"):
        gr.basic_block.__init__(
            self,
            name='AES-128 CTR Encoder (fixed)',
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        self.key_hex = key_hex
        self.nonce_hex = nonce_hex
        self._init_cipher()

    def _init_cipher(self):
        key = bytes.fromhex(self.key_hex)
        nonce = bytes.fromhex(self.nonce_hex)
        if len(key) != 16:
            raise ValueError("key_hex must decode to exactly 16 bytes (32 hex chars)")
        if len(nonce) != 16:
            raise ValueError("nonce_hex must decode to exactly 16 bytes (32 hex chars)")
        ctr = Counter.new(128, initial_value=int.from_bytes(nonce, 'big'))
        self.cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    def forecast(self, noutput_items, ninputs):
        return [noutput_items for _ in range(ninputs)]

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        n_out = min(len(in0), len(out))
        if n_out > 0:
            encrypted = self.cipher.encrypt(in0[:n_out].tobytes())
            out[:n_out] = np.frombuffer(encrypted, dtype=np.uint8)
        self.consume(0, n_out)
        return n_out
