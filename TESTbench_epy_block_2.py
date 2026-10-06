"""
embedded python block -- Reed-Solomon FEC Encoder WITH PREAMBLE (Stage B)
Replaces epy_block_2 in TESTbench for Phase 1, Stage B.

Change vs. the current block: before every 72-byte codeword it emits the
2-byte sync word 0xEB 0x90. The receiver (rx_decode.py) searches for this
constant pattern, so RS codeword framing no longer has to be brute-forced,
and frame start is unambiguous.

Paste this whole file into the epy_block_2 "Source Code" tab in GRC.
Parameters stay identical: k=56, nsym=16.

Frame layout on the air (before unpack_k_bits):
    [0xEB 0x90][codeword: 56 message bytes + 16 parity bytes]  = 74 bytes
"""

import numpy as np
from gnuradio import gr
from reedsolo import RSCodec


class blk(gr.basic_block):
    def __init__(self, k=56, nsym=16):
        gr.basic_block.__init__(
            self,
            name='Reed-Solomon FEC Encoder + preamble',
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        self.k = k
        self.nsym = nsym
        if self.k + self.nsym > 255:
            raise ValueError("k + nsym must be <= 255 (GF(2^8) limit)")
        self.rsc = RSCodec(nsym)
        self.buf = bytearray()
        self.preamble = b'\xEB\x90'

    def forecast(self, noutput_items, ninputs):
        return [1 for _ in range(ninputs)]

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        self.buf.extend(in0.tobytes())
        self.consume(0, len(in0))

        out = output_items[0]
        produced = 0
        n = self.k + self.nsym
        frame = len(self.preamble) + n
        while len(self.buf) >= self.k and produced + frame <= len(out):
            chunk = bytes(self.buf[:self.k])
            del self.buf[:self.k]
            encoded = self.rsc.encode(chunk)
            data = self.preamble + encoded
            out[produced:produced + frame] = np.frombuffer(data, dtype=np.uint8)
            produced += frame
        return produced