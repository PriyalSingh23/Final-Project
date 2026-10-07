"""
lpi_grc.py -- GNU Radio integration for ShadowComm LPI v4 (no OOT module needed)
================================================================================

Two small, framework-free cores live here:

    TxCore : (128 BPSK symbols per frame)  -> (576 complex IQ samples: pilot + spread frame)
    RxCore : stream of complex IQ samples  -> keyed-pilot sync -> bits -> RS/AES -> text

They are plain Python (numpy + torch).  The `.grc` flowgraphs embed a 15-line
`epy_block` that only *delegates* to these classes, so:

  * the same code runs in GRC, in `python rx_usrp.py`, and in `pytest`
    (`python lpi_grc.py --selftest` needs **no** GNU Radio and **no** radio),
  * a GRC bug can never change the PHY -- and vice versa.

Used by tx_lpi_v4.grc / rx_lpi_v4.grc.  Set the `lpi_dir` parameter of the block
(or export LPI_V4_DIR) to the folder that contains lpi_core.py + the .pt files.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

LPI_DIR = os.environ.get("LPI_V4_DIR", _HERE)


def _load(cfg_json: str, ckpt: str, device: str = "cpu"):
    import torch
    from lpi_core import LPIConfig, LPIGenerator, LPIDecoder, load_config
    cfg = load_config(cfg_json) if cfg_json and os.path.exists(cfg_json) else LPIConfig()
    gen = LPIGenerator(cfg).to(device)
    dec = LPIDecoder(cfg).to(device)
    if ckpt and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(blob, dict):
            gen.load_state_dict(blob.get("generator", {}), strict=False)
            dec.load_state_dict(blob.get("decoder", {}), strict=False)
        else:
            gen.load_state_dict(blob, strict=False)
    dec.tie_to(gen)
    gen.eval(); dec.eval()
    torch.set_grad_enabled(False)
    return cfg, gen, dec


class TxCore:
    """BPSK symbol vector (+/-1) -> IQ frame with the keyed pilot in front.

    The GRC side feeds this from `digital.chunks_to_symbols_bc` with constellation
    [-1, +1], i.e. bit 1 -> -1 and bit 0 -> +1 is irrelevant: we threshold the real
    part, so either mapping works as long as it is bipolar.
    """

    def __init__(self, ckpt: str = "", cfg_json: str = "", key: str = "SESSION-KEY",
                 amp: float = 0.05, device: str = "cpu"):
        import torch
        self.torch = torch
        self.cfg, self.gen, self.dec = _load(cfg_json, ckpt, device)
        self.key = key
        self.amp = float(amp)
        from lpi_core import make_preamble
        self.pilot = make_preamble(key, self.cfg.pilot_len)
        self.T = self.cfg.frame_len
        self.nb = self.cfg.n_bits
        self.L = self.T + self.cfg.pilot_len
        self.z = torch.randn(1, self.cfg.z_dim)
        self.log = []

    # out vector length = L, in vector length = n_bits
    def in_len(self) -> int:
        return self.nb

    def out_len(self) -> int:
        return self.L

    def work(self, in0: np.ndarray, out: np.ndarray) -> int:
        """in0: (nvec, n_bits) complex, out: (nvec, L) complex."""
        torch = self.torch
        n = in0.shape[0]
        b = (in0.real.astype(np.float32) > 0).astype(np.float32) * 2 - 1     # {-1,+1}
        z = torch.randn(n, self.cfg.z_dim)
        with torch.no_grad():
            x = self.gen(z, torch.from_numpy(b))                 # (n, 2, T)
        from lpi_core import unit_power
        x = unit_power(x).cpu().numpy()
        pay = x[:, 0] + 1j * x[:, 1]
        from lpi_core import frame_with_pilot
        iq = frame_with_pilot(pay, self.pilot).reshape(n, self.L)
        m = np.abs(iq).max() + 1e-12
        iq = iq * (self.amp / m)
        out[: n] = iq.astype(np.complex64)
        return n


class RxCore:
    """Accumulate IQ, lock on the keyed pilot, decode, print/emit text.

    Parameters
    ----------
    burst_frames : frames per AES/RS burst (3 at n_bits=128) -- used to know when a
        complete message is available.
    window : samples to buffer per work call; longer = better CFO estimate, more delay.
    """

    def __init__(self, ckpt: str = "", cfg_json: str = "", key: str = "SESSION-KEY",
                 mode: str = "ctr", window: int = 0, device: str = "cpu",
                 verbose: bool = True, out_file: str = ""):
        self.cfg, self.gen, self.dec = _load(cfg_json, ckpt, device)
        self.key, self.mode, self.verbose = key, mode, verbose
        self.device = device
        from lpi_core import make_preamble
        self.pilot = make_preamble(key, self.cfg.pilot_len)
        from lpi_crypto import FrameCodec
        self.fc = FrameCodec(key, self.cfg.n_bits, mode=mode)
        self.L = self.cfg.frame_len + self.cfg.pilot_len
        self.window = int(window) or (8 * self.L + int(self.cfg.fs))
        self.buf = np.zeros(0, np.complex64)
        self.out_file = out_file
        self.last = {}

    def work(self, in0: np.ndarray) -> int:
        """in0: (N,) complex.  Buffers, decodes when a full burst is available."""
        self.buf = np.concatenate([self.buf, in0.astype(np.complex64)])
        need = self.fc.frames * self.L                       # samples for one burst
        if self.buf.size >= max(need, self.window):
            self._decode(self.buf)
            self.buf = self.buf[-(self.L * 2):]              # keep a little overlap
        return in0.size

    def _decode(self, raw: np.ndarray):
        t0 = time.perf_counter()
        res = decode_capture(raw, self.cfg, self.dec, self.key, self.mode,
                             n_frames=4096, device=self.device,
                             auto_recover=True)
        res["dt_s"] = time.perf_counter() - t0
        self.last = res
        if self.verbose:
            if not res.get("n_frames"):
                print(f"[lpi-rx] no lock ({raw.size} samples, "
                      f"{raw.size / self.cfg.fs:.2f} s, {res['dt_s'] * 1e3:.0f} ms) "
                      f"-- {res.get('note', '')}")
            else:
                print(f"[lpi-rx] {res['n_frames']} frames | CFO "
                      f"{res.get('cfo_hz', 0):+.1f} Hz (coarse "
                      f"{res.get('coarse_hz', 0.0):+.1f}) | pilot "
                      f"{res.get('pilot_snr_db', float('nan')):.1f} dB | "
                      f"{res['dt_s'] * 1e3:.0f} ms | CRC {res.get('crc_ok', '?')} | "
                      f"{res.get('text', '')[:70]!r}")
        if self.out_file:
            with open(self.out_file, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} frames={res.get('n_frames', 0)} "
                        f"cfo={res.get('cfo_hz', 0.0):.1f} "
                        f"coarse={res.get('coarse_hz', 0.0):.1f} "
                        f"pilot={res.get('pilot_snr_db', float('nan')):.1f} "
                        f"crc={'OK' if res.get('ok') else 'BAD'} "
                        f"msg={res.get('text', '')}\n")
        return res


# ------------------------------------------------------------------ wrappers ----
def make_grc_sources() -> tuple[str, str]:
    """The two `_source` strings embedded in the .grc files (kept here so they can
    be syntax-checked without opening GNU Radio Companion)."""
    tx = '''import numpy as np
from gnuradio import gr
import os, sys
LPI_DIR = os.path.expanduser(lpi_dir)
if LPI_DIR not in sys.path:
    sys.path.insert(0, LPI_DIR)
import lpi_grc as _g


class lpi_tx(gr.sync_block):
    """BPSK symbol vector (n_bits) -> LPI IQ frame (pilot + spread payload)."""

    def __init__(self, lpi_dir="", ckpt="", cfg_json="", key="SESSION-KEY", amp=0.05):
        core = _g.TxCore(ckpt, cfg_json, key, amp)
        gr.sync_block.__init__(self, name="lpi_tx",
                               in_sig=[(np.complex64, core.in_len())],
                               out_sig=[(np.complex64, core.out_len())])
        self.core = core

    def work(self, input_items, output_items):
        return self.core.work(input_items[0], output_items[0])
'''
    rx = '''import numpy as np
from gnuradio import gr
import os, sys
LPI_DIR = os.path.expanduser(lpi_dir)
if LPI_DIR not in sys.path:
    sys.path.insert(0, LPI_DIR)
import lpi_grc as _g


class lpi_rx(gr.sync_block):
    """IQ stream -> keyed pilot sync -> RS/AES -> text (printed + file)."""

    def __init__(self, lpi_dir="", ckpt="", cfg_json="", key="SESSION-KEY",
                 mode="ctr", out_file="", verbose=True, window=0):
        core = _g.RxCore(ckpt, cfg_json, key, mode, window, "cpu", verbose, out_file)
        gr.sync_block.__init__(self, name="lpi_rx", in_sig=[np.complex64], out_sig=[])
        self.core = core
        self.set_relative_rate(1.0)

    def work(self, input_items, output_items):
        return self.core.work(input_items[0])
'''
    return tx, rx


# --------------------------------------------------------------------- decode ----
def decode_capture(raw, cfg, dec, key: str = "SESSION-KEY", mode: str = "ctr",
                   n_frames: int = 4096, device: str = "cpu", auto_recover: bool = True):
    """One shared field-receive path for rx_usrp.py, the GRC block and the tests.

    Pilot sync -> soft decode -> RS/AES text.  If the burst does *not* come back
    clean we do not give up and we do not guess: we run the coarse CFO acquisition
    (which is unambiguous, unlike the +-fs/(2*T_frame) = +-213 Hz pilot-only fine
    estimate) and try again.  That single retry is what makes two free-running
    B210 LOs (~5 kHz apart at 2.484 GHz) a non-event on the bench.
    """
    from lpi_core import LockNotFound, coarse_acquisition, frames_to_bits, \
        make_preamble, sync_and_correct
    from lpi_crypto import FrameCodec

    pl = make_preamble(key, cfg.pilot_len)
    fc = FrameCodec(key, cfg.n_bits, mode=mode)
    raw = np.asarray(raw, np.complex64).reshape(-1)
    stride = fc.frames * cfg.n_bits          # samples worth of bits per AES/RS burst

    def attempt(coarse_hz):
        try:
            frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n_frames,
                                            cfo_search=True, return_info=True,
                                            coarse_hz=coarse_hz)
        except LockNotFound as e:
            return None, {"note": str(e), "n_frames": 0, "coarse_hz": float(coarse_hz),
                          "cfo_hz": float(coarse_hz), "pilot_snr_db": float("nan"),
                          "dropped_frames": 0, "offset": 0, "text": "", "ok": False,
                          "bursts": 0, "crc_ok": "0/0", "ber": 1.0}
        out = dict(info)
        out["coarse_hz"] = float(coarse_hz)
        if frames.shape[0] == 0:
            out.update(text="", ok=False, bursts=0, crc_ok="0/0", ber=1.0)
            return frames, out
        bits = frames_to_bits(dec, frames, device)
        out["n_frames_decoded"] = int(bits.size // cfg.n_bits)
        texts, crcs = [], []
        for i in range(0, bits.size - stride + 1, stride):
            t, ok = fc.decode(bits[i:i + stride])
            texts.append(t)
            crcs.append(ok)
        out["text"] = "".join(texts)
        out["bursts"] = len(texts)
        out["crc_ok"] = f"{sum(crcs)}/{len(crcs)}" if texts else "0/0"
        out["ok"] = bool(texts) and all(crcs)
        out["ber"] = float("nan")
        return frames, out

    frames, res = attempt(0.0)
    if auto_recover and not res.get("ok", False):
        # not clean -> acquire the offset properly and retry (see coarse_acquisition)
        ch = coarse_acquisition(raw, cfg, pl, key, dec=dec, n_frames=12)
        if abs(ch) > 1e-9:
            f2, r2 = attempt(ch)
            better = r2.get("ok", False) or (
                (f2 is not None and f2.shape[0] > (0 if frames is None else frames.shape[0]))
                and not res.get("text"))
            if better:
                frames, res = f2, r2
    res["need_frames"] = fc.frames
    return res


# ------------------------------------------------------------------- selftest ----
def _selftest(ckpt, cfg_json, key, snr_db=8.0, nmsg=24, cfo_max=150.0):
    """No GNU Radio, no radio: TxCore -> channel -> RxCore, text in == text out."""
    import torch
    from lpi_core import apply_channel
    from lpi_crypto import FrameCodec
    print(f"[grc-selftest] TxCore/RxCore loopback over the modelled channel "
          f"(SNR {snr_db} dB, {nmsg} bursts, ckpt={ckpt or 'random'})")
    tx = TxCore(ckpt, cfg_json, key)
    rx = RxCore(ckpt, cfg_json, key, verbose=False)
    tx.cfg.cfo_max = float(cfo_max)          # the pilot fine sync's unambiguous range
    rx.cfg.cfo_max = float(cfo_max)
    fc = FrameCodec(key, tx.nb)
    msg = "SHADOWCOMM-v4-GRC-LOOPBACK"[: fc.max_len]
    bits = np.concatenate([fc.encode(msg) for _ in range(nmsg)])
    nf = bits.size // tx.nb
    vin = np.zeros((nf, tx.nb), np.complex64)
    vin.real = bits.reshape(nf, tx.nb).astype(np.float32) * 2 - 1
    vout = np.zeros((nf, tx.L), np.complex64)
    tx.work(vin, vout)
    iq = vout.reshape(-1)
    iq = iq / (np.sqrt((iq.real ** 2 + iq.imag ** 2).mean()) + 1e-12)   # unit power
    L1 = tx.L
    t = torch.from_numpy(np.stack([iq.real, iq.imag])[None].astype(np.float32))
    n = (t.shape[-1] - 1) // L1 + 1
    t = torch.nn.functional.pad(t, (0, n * L1 - t.shape[-1])).view(1, 2, n, L1).flatten(2)
    y, _ = apply_channel(t, tx.cfg, snr_db=snr_db, impairments=True)
    y = y[:, :, : iq.size]
    raw = (y[0, 0].numpy() + 1j * y[0, 1].numpy()).astype(np.complex64)
    res = rx._decode(raw)
    ok = bool(res) and res.get("ok")
    print(f"[grc-selftest] frames={res.get('n_frames')} cfo={res.get('cfo_hz', 0):+.1f} Hz "
          f"pilot={res.get('pilot_snr_db', float('nan')):.1f} dB text={res.get('text','')[:40]!r} "
          f"-> {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "run", "lpi_v4.best.pt"))
    ap.add_argument("--config", default=os.path.join(_HERE, "run", "lpi_v4.best.json"))
    ap.add_argument("--key", default="SESSION-KEY")
    ap.add_argument("--snr", type=float, default=8.0)
    ap.add_argument("--bursts", type=int, default=24)
    ap.add_argument("--cfo-max", type=float, default=150.0,
                    help="CFO half-range [Hz] for the synthetic channel. Beyond "
                         "+-fs/(2*frame_period) the pilot fine sync aliases and the "
                         "coarse acquisition has to run (it does, automatically).")
    ap.add_argument("--print-sources", action="store_true")
    a = ap.parse_args()
    if a.print_sources:
        for s in make_grc_sources():
            print(s)
            print("# " + "-" * 70)
        raise SystemExit(0)
    raise SystemExit(_selftest(a.ckpt, a.config, a.key, a.snr, a.bursts, a.cfo_max))
