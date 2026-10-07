#!/usr/bin/env python3
"""
tx_usrp.py -- real-time LPI transmit for Ettus USRP (B200/B210/X310/N210).
==========================================================================

    text -> AES-128-CTR(+CRC-8, RS(42,34)) -> keyed spread frames -> CGAN
    waveform (statistically noise) -> UHD -> antenna

This is the recommended field path: raw `uhd` has no GNU Radio startup cost, and
it uses the *same* numpy/torch code as the trainer and the evaluator, so a
bench-simulated result and an over-the-air result cannot silently disagree.

    python tx_usrp.py --text "ALPHA-INDIA-001" --bursts 20
    python tx_usrp.py --addr 192.168.10.2 --freq 2.484e9 --rate 245760 --gain 0
    python tx_usrp.py --loop --period 0.2          # keep transmitting
    python tx_usrp.py --file tx.npz                # no radio: write samples for RX

Practical notes
---------------
* B200/B210 cannot go below ~208 kS/s, so the default rate is 245760 S/s
  (= 61.44 MHz / 250, exact integer decimation -> no fractional clock drift).
  The "32 kS/s" line in the old checklist is not reachable on a B210; the
  probability of intercept is still low because a frame carries only 128 bits
  over 576 samples (9.0 dB processing gain) at -X dB SNR inside the noise floor.
* --amp sets the fraction of full scale that leaves the DAC (0.05 default).
  Start at 0.02-0.05 and only raise it until the receiver locks: TX gain is a
  link-budget knob, not a stealth knob.
* --key must match the receiver's.  The keyed Zadoff-Chu preamble is what the
  receiver locks on; a wrong key costs ~everything.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpi_core import (LPIConfig, LPIGenerator, frame_with_pilot, load_config,
                      make_preamble, unit_power)
from lpi_crypto import FrameCodec


def build_iq(cfg: LPIConfig, gen: LPIGenerator, bits: np.ndarray, key: str,
             device="cpu", seed: int = 0) -> np.ndarray:
    """bits (N*n_bits,) uint8 -> (N*(frame+pilot),) complex64 at unit mean power."""
    n = bits.size // cfg.n_bits
    b = torch.from_numpy(bits[: n * cfg.n_bits].reshape(n, cfg.n_bits).astype(np.float32)
                         * 2 - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, cfg.z_dim, generator=g)
    gen.eval()
    with torch.no_grad():
        x = unit_power(gen(z.to(device), b.to(device))).cpu().numpy()   # (n,2,T)
    pay = (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
    return frame_with_pilot(pay, make_preamble(key, cfg.pilot_len)).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="", help='USRP address, e.g. 192.168.10.2 (""=first local)')
    ap.add_argument("--text", default="LPI-v4 SHADOWCOMM OVER-THE-AIR TEST 001")
    ap.add_argument("--key", default="SESSION-KEY", help="session key (must match RX)")
    ap.add_argument("--mode", default="ctr", choices=["ctr", "gcm"])
    ap.add_argument("--bursts", type=int, default=4, help="AES/RS bursts per transmission")
    ap.add_argument("--period", type=float, default=0.0, help="seconds between bursts")
    ap.add_argument("--loop", action="store_true", help="transmit forever (Ctrl-C stops)")
    ap.add_argument("--rate", type=float, default=245760.0)
    ap.add_argument("--freq", type=float, default=2.484e9)
    ap.add_argument("--gain", type=float, default=0.0, help="TX front-end gain [dB]")
    ap.add_argument("--ant", default="")
    ap.add_argument("--clock-source", default="", help="'external' / 'gpsdo'")
    ap.add_argument("--amp", type=float, default=0.05, help="peak amplitude, fraction of FS")
    ap.add_argument("--file", default="", help="no radio: save tx.npz (RX reads it with --capture)")
    ap.add_argument("--ckpt", default="run/lpi_v4.best.pt")
    ap.add_argument("--config", default="run/lpi_v4.best.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = LPIGenerator(cfg).to(device)
    if os.path.exists(args.ckpt):
        blob = torch.load(args.ckpt, map_location=device, weights_only=False)
        gen.load_state_dict(blob.get("generator", blob) if isinstance(blob, dict) else blob,
                            strict=False)
        print(f"[tx] generator: {args.ckpt} (epoch {blob.get('epoch','?') if isinstance(blob,dict) else '?'})")
    else:
        print(f"[tx] WARNING: no checkpoint at {args.ckpt} -> untrained waveform")

    fc = FrameCodec(args.key, cfg.n_bits, mode=args.mode)
    burst = fc.encode_long(args.text)
    bits = np.tile(burst, max(1, args.bursts))
    iq = build_iq(cfg, gen, bits, args.key, device, seed=int(time.time()) & 0xFFFF)
    iq = (iq * (args.amp / (np.abs(iq).max() + 1e-12))).astype(np.complex64)

    L = cfg.pilot_len + cfg.frame_len
    print(f"[tx] {args.text[:40]!r} -> {burst.size} bits -> {iq.size} samples "
          f"= {iq.size/cfg.fs*1e3:.1f} ms ({iq.size//L} frames of {L} samples)")
    print(f"[tx] per burst: {burst.size} coded bits over {fc.frames} frames "
          f"({fc.frames*cfg.n_bits} channel bits) | on-air bit rate "
          f"{cfg.n_bits*fc.frames/(len(burst)//8*0+ (fc.frames*(cfg.pilot_len+cfg.frame_len)/cfg.fs)):.0f} bit/s")

    if args.file:
        np.savez_compressed(args.file, iq=iq, bits=bits, fs=cfg.fs, freq=args.freq,
                            key=args.key, n_bits=cfg.n_bits, frame_len=cfg.frame_len,
                            pilot_len=cfg.pilot_len, text=args.text)
        print(f"[tx] wrote {args.file}")
        return 0

    from uhd_io import Radio, find_devices
    try:
        rad = Radio(addr=args.addr, rate=args.rate, freq=args.freq, gain=args.gain,
                    ant=args.ant, clock=args.clock_source, direction="tx")
    except SystemExit:
        print(find_devices())
        return 2
    print(f"[tx] {rad.summary()}")
    n = 0
    try:
        while True:
            rad.send(iq)
            n += iq.size
            print(f"\r[tx] sent {n} samples ({n/rad.rate:.2f} s)", end="", flush=True)
            if not args.loop:
                break
            if args.period > 0:
                time.sleep(args.period)
    except KeyboardInterrupt:
        print("\n[tx] stopped by user")
    print(f"\n[tx] done: {n} samples = {n/rad.rate:.3f} s of radiation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
