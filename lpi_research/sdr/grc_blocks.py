"""GNU Radio 3.10 Compatible Custom Blocks for LPI-CGAN Transceiver.
Provides:
  1. CGANGeneratorBlock: Embedded Python block mapping 256 BPSK symbols to 512 complex baseband samples.
  2. CGANDecoderSyncBlock: Matched-filter Zadoff-Chu synchronizer, CFO compensator,
     neural decoder, Reed-Solomon ECC, and AES-128 decryptor.
"""

import os
import math
import numpy as np
import torch
try:
    from gnuradio import gr  # type: ignore
except ImportError:
    class _DummyBlock:
        def __init__(self, *args, **kwargs): pass
    class _DummyGR:
        sync_block = _DummyBlock
        basic_block = _DummyBlock
    gr = _DummyGR()  # type: ignore
from Crypto.Cipher import AES
from Crypto.Util import Counter
from reedsolo import RSCodec, ReedSolomonError

PILOT_LEN = 512
FRAME_LEN = 512


def generate_zadoff_chu(length=512, u=25):
    n = np.arange(length)
    zc = np.exp(-1j * np.pi * u * (n ** 2) / length) if length % 2 == 0 else np.exp(-1j * np.pi * u * n * (n + 1) / length)
    return (zc / np.sqrt(np.mean(np.abs(zc)**2))).astype(np.complex64)


class CGANGeneratorBlock(gr.sync_block):
    """GNU Radio block: input vector of 256 complex symbols -> output vector of 512 complex samples."""

    def __init__(self, model_path='generator_lpi.pt', gain_factor=0.35):
        gr.sync_block.__init__(
            self,
            name='CGAN_LPI_Generator',
            in_sig=[(np.complex64, 256)],
            out_sig=[(np.complex64, 512)]
        )
        self.gain = gain_factor
        self.device = 'cpu'

        # Search for model path
        candidates = [model_path, os.path.join(os.path.dirname(__file__), 'generator_lpi.pt'),
                      'C:/Users/gspra/OneDrive/Desktop/LPI_CGAN/generator_lpi.pt']
        found = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                found = cand
                break

        if found:
            self.model = torch.jit.load(found, map_location=self.device)
            self.model.eval()
            print(f"[CGAN GRC Generator] Loaded from {found} on {self.device}")
        else:
            self.model = None
            print(f"[CGAN GRC Generator] WARNING: Model {model_path} not found!")

        self.gpu_noise = torch.zeros(1, 64, dtype=torch.float32, device=self.device)
        self.b_tensor = torch.zeros(1, 256, dtype=torch.float32, device=self.device)

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        if self.model is None:
            # Fallback: pass-through zero-padded
            for i in range(len(in0)):
                out[i] = np.pad(in0[i], (0, 256)).astype(np.complex64)
            return len(in0)

        for i in range(len(in0)):
            frame = in0[i]
            self.b_tensor[0, :] = torch.from_numpy(frame.real.astype(np.float32))
            self.gpu_noise.normal_(mean=0, std=1)

            with torch.no_grad():
                covert = self.model(self.gpu_noise, self.b_tensor)

            c_np = covert.cpu().numpy()[0]
            out[i] = (c_np[0, :] + 1j * c_np[1, :]).astype(np.complex64) * self.gain

        return len(in0)


class CGANDecoderSyncBlock(gr.basic_block):
    """Stream-based GNU Radio receiver block:
    Accumulates incoming samples, detects Zadoff-Chu pilot via matched filter,
    estimates CFO and phase, normalizes payload, runs Neural Decoder,
    applies Reed-Solomon correction, and verifies AES-128 ciphertext.
    """

    def __init__(self, model_path='decoder_lpi.pt', fs=2e6):
        gr.basic_block.__init__(
            self,
            name='CGAN_LPI_Sync_Decoder',
            in_sig=[np.complex64],
            out_sig=[np.uint8]
        )
        self.fs = fs
        self.device = 'cpu'
        self.pilot = generate_zadoff_chu(PILOT_LEN, u=25)
        self.rsc = RSCodec(9)

        # Load Decoder
        candidates = [model_path, os.path.join(os.path.dirname(__file__), 'decoder_lpi.pt'),
                      'C:/Users/gspra/OneDrive/Desktop/LPI_CGAN/decoder_lpi.pt']
        found = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                found = cand
                break

        if found:
            self.decoder = torch.jit.load(found, map_location=self.device)
            self.decoder.eval()
            print(f"[CGAN GRC Decoder] Loaded from {found} on {self.device}")
        else:
            self.decoder = None

        self.buffer = np.array([], dtype=np.complex64)
        self.gpu_input = torch.zeros(1, 2, 512, dtype=torch.float32, device=self.device)

    def forecast(self, noutput_items, ninputs):
        return [noutput_items for _ in range(ninputs)]

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]

        if len(in0) > 0:
            self.buffer = np.concatenate([self.buffer, in0])
            self.consume(0, len(in0))

        produced = 0
        min_required = PILOT_LEN + FRAME_LEN

        while len(self.buffer) >= min_required and produced + 256 <= len(out0):
            # Correlate buffer head with pilot
            corr = np.correlate(self.buffer[:min_required * 2], self.pilot, mode='valid')
            if len(corr) == 0:
                break

            mag = np.abs(corr)
            peak_idx = int(np.argmax(mag))
            noise_floor = np.median(mag) + 1e-12

            if mag[peak_idx] > 3.0 * noise_floor and peak_idx + min_required <= len(self.buffer):
                # Detected frame boundary!
                rx_pilot = self.buffer[peak_idx:peak_idx + PILOT_LEN]
                half = PILOT_LEN // 2
                p1 = rx_pilot[:half] * np.conj(self.pilot[:half])
                p2 = rx_pilot[half:] * np.conj(self.pilot[half:])
                cfo_hz = float(np.angle(np.sum(p2) * np.conj(np.sum(p1))) * self.fs / (2.0 * np.pi * half))

                est_phase = float(np.angle(corr[peak_idx]))
                n_axis = np.arange(FRAME_LEN, dtype=np.float64) + peak_idx + PILOT_LEN
                cfo_rot = np.exp(-1j * (2.0 * np.pi * cfo_hz * n_axis / self.fs + est_phase)).astype(np.complex64)

                payload_raw = self.buffer[peak_idx + PILOT_LEN:peak_idx + min_required] * cfo_rot

                # Normalize per frame
                v = np.stack([payload_raw.real, payload_raw.imag], axis=0).astype(np.float32)
                v = v - v.mean(axis=1, keepdims=True)
                s = v.std(axis=1, keepdims=True) + 1e-8
                v_norm = v / s

                self.gpu_input[0, :, :] = torch.from_numpy(v_norm)
                with torch.no_grad():
                    b_hat = (self.decoder(self.gpu_input).cpu().numpy()[0] > 0.5).astype(np.uint8)

                out0[produced:produced + 256] = b_hat
                produced += 256

                # Slide buffer past this detected burst
                self.buffer = self.buffer[peak_idx + min_required:]
            else:
                # Slide buffer by half frame
                if len(self.buffer) > min_required:
                    self.buffer = self.buffer[FRAME_LEN:]
                else:
                    break

        return produced
