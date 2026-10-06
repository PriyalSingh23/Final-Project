"""
embedded python block -- CORRECTED VERSION
Reed-Solomon FEC Decoder for GNU Radio

*** REPLACES the earlier fec_rs_decoder.py from Stage 3 -- same underlying
bug and same fix as fec_rs_encoder_v2.py; see that file's header for the
full explanation. ***

MUST use the same k and nsym as the encoder.
"""

import numpy as np
from gnuradio import gr
from reedsolo import RSCodec, ReedSolomonError


class blk(gr.basic_block):
    def __init__(self, k=56, nsym=16):
        gr.basic_block.__init__(
            self,
            name='Reed-Solomon FEC Decoder (corrected)',
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        self.k = k
        self.nsym = nsym
        self.n = k + nsym
        self.rsc = RSCodec(nsym)
        self.buf = bytearray()
        self.blocks_ok = 0
        self.blocks_corrected = 0
        self.blocks_failed = 0

    def forecast(self, noutput_items, ninputs):
        return [1 for _ in range(ninputs)]

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        self.buf.extend(in0.tobytes())
        self.consume(0, len(in0))

        out = output_items[0]
        produced = 0
        while len(self.buf) >= self.n and produced + self.k <= len(out):
            codeword = bytes(self.buf[:self.n])
            del self.buf[:self.n]
            try:
                decoded, _, errata = self.rsc.decode(codeword)
                if len(errata) > 0:
                    self.blocks_corrected += 1
                    print(f"[RS decoder] corrected {len(errata)} byte error(s)")
                else:
                    self.blocks_ok += 1
                out[produced:produced + self.k] = np.frombuffer(bytes(decoded), dtype=np.uint8)
            except ReedSolomonError as e:
                self.blocks_failed += 1
                print(f"[RS decoder] UNCORRECTABLE block ({e}) -- passing through uncorrected")
                out[produced:produced + self.k] = np.frombuffer(codeword[:self.k], dtype=np.uint8)
            produced += self.k
        return produced
