#!/usr/bin/env python3
"""
rx_usrp.py -- LPI receive + decode (over the air, over cable, or from a file).
==============================================================================

    IQ capture -> keyed-pilot sync (timing + CFO + phase) -> matched filter /
    ZF + neural refinement -> bits -> RS(42,34) + CRC-8 + AES-128 -> text

    python rx_usrp.py --addr 192.168.10.2 --seconds 0.2            # live, single shot
    python rx_usrp.py --watch                                      # keep decoding bursts
    python rx_usrp.py --capture cap.npz                            # offline file
    python rx_usrp.py --uhd-info                                   # what's on the bench?

Why it is built this way
------------------------
The synchroniser is the *same* function the trainer evaluates and that the GNU
Radio block calls (lpi_core.sync_and_correct), so if `--selftest` passes on the
laptop, the only variables on the bench are frequency, gain and clock offset.
It reports: frames locked, CFO estimate, pilot SNR, residual BER and the
decoded text with its CRC verdict -- five numbers that tell you which of the
four problems you actually have:

  no frames found ......... wrong centre freq / rate, or TX below the noise floor
  frames but pilot SNR <8dB  gain/antenna problem, or key mismatch
  pilot SNR fine, BER high . CFO beyond +-300 Hz, sample-clock offset, clipping
  BER tiny, CRC bad ....... key/mode mismatch or an odd number of frames captured
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpi_core import (LPIDecoder, LockNotFound, coarse_acquisition, frame_with_pilot,
                      load_config, make_preamble, sync_and_correct)
from lpi_crypto import FrameCodec


def load_rx(cfg, ckpt, device):
    dec = LPIDecoder(cfg).to(device)
    from lpi_core import LPIGenerator
    gen = LPIGenerator(cfg).to(device)          # the decoder is tied to G's codebook
    if os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(blob, dict):
            if "generator" in blob:
                gen.load_state_dict(blob["generator"], strict=False)
            if "decoder" in blob:
                dec.load_state_dict(blob["decoder"], strict=False)
            print(f"[rx] weights: {ckpt} (epoch {blob.get('epoch','?')})")
        else:
            print(f"[rx] weights: {ckpt} (bare state dict)")
            dec.load_state_dict(blob, strict=False)
    else:
        print(f"[rx] WARNING no checkpoint at {ckpt}: bits will be noise")
    dec.tie_to(gen)
    dec.eval()
    return dec


def decode_capture(raw, cfg, dec, key, mode, n_frames, device, coarse="auto",
                   span=6000.0):
    """Sync + decode one capture.

    `coarse` controls the LO-offset acquisition retry that two free-running radios
    make necessary:  "auto" -> search only when the direct lock is missing/weak,
    "off" -> never,  a number -> always search +/- that many Hz.  The per-frame
    pilot derotation is only unambiguous to +/-(fs/2/frame_period) (~213 Hz here),
    so a 2 ppm LO error at 2.484 GHz (~5 kHz) *must* come from this search.
    """
    pl = make_preamble(key, cfg.pilot_len)
    raw = np.asarray(raw, np.complex64).reshape(-1)
    c = str(coarse).strip().lower()
    span_hz = span if c in ("auto", "on", "") else (0.0 if c == "off" else float(c))
    ch = 0.0
    err = ""
    frames = None
    info: dict = {}
    try:
        frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n_frames,
                                        cfo_search=True, return_info=True)
    except LockNotFound as e:
        err = str(e)
        if c == "off":
            raise
    weak = frames is None or frames.shape[0] < 2 or info.get("pilot_snr_db", 0.0) < 8.0
    if c != "off" and (weak or c not in ("auto", "on", "")):
        ch = coarse_acquisition(raw, cfg, pl, key, dec=dec, span=span_hz, n_frames=12)
        if ch:
            try:
                frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n_frames,
                                                cfo_search=True, return_info=True,
                                                coarse_hz=ch)
            except LockNotFound as e:
                err, frames = str(e), None
    if frames is None or frames.shape[0] == 0:
        return {"offset": 0, "n_frames": 0, "coarse_hz": ch, "cfo_hz": ch,
                "pilot_snr_db": float("nan"), "dropped_frames": 0, "text": "",
                "ok": False, "ber": 1.0, "searched_hz": span_hz,
                "note": err or "no lock"}

    from lpi_core import frames_to_bits
    out = dict(info)
    bits = frames_to_bits(dec, frames, device)
    out["n_frames_decoded"] = int(bits.size // cfg.n_bits)
    fc = FrameCodec(key, cfg.n_bits, mode=mode)          # one burst = fc.frames frames
    stride = fc.frames * cfg.n_bits
    texts, crcs = [], []
    for i in range(0, bits.size - stride + 1, stride):
        t, ok = fc.decode(bits[i:i + stride])
        texts.append(t)
        crcs.append(ok)
    out["text"] = "".join(texts)
    out["ok"] = bool(texts) and all(crcs)
    out["bursts"] = len(texts)
    out["crc_ok"] = f"{sum(crcs)}/{len(crcs)}" if texts else "0/0"
    out["ber"] = float("nan")
    return out


def print_result(res, t_take):
    if not res.get("n_frames", 0):
        where = res.get("searched", "capture")
        note = res.get("note", "")
        print(f"  [{t_take:.2f} s] NO LOCK in {where} -- {note}")
        print("           try: --coarse 8000 (LO offset), higher --gain, "
              "same --rate/--freq as TX, --key match")
        return
    print(f"  [{t_take:.2f} s] frames {res.get('n_frames', 0):>4} | offset "
          f"{res.get('offset', 0):>6} | CFO {res.get('cfo_hz', 0):+9.1f} Hz "
          f"(coarse {res.get('coarse_hz', 0.0):+8.1f}) | pilot-SNR "
          f"{res.get('pilot_snr_db', float('nan')):5.1f} dB | dropped "
          f"{res.get('dropped_frames', 0)}")
    if res.get("bursts"):
        print(f"           bursts {res['bursts']} crc {res.get('crc_ok','?')} "
              f"{'OK' if res.get('ok') else 'FAIL'} -> {res['text'][:80]!r}")
    else:
        print(f"           locked but no complete burst (need >= {res.get('need','?')} "
              f"consecutive frames of {res.get('frame_len','?')}+pilot)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="")
    ap.add_argument("--seconds", type=float, default=0.25)
    ap.add_argument("--rate", type=float, default=245760.0)
    ap.add_argument("--freq", type=float, default=2.484e9)
    ap.add_argument("--gain", type=float, default=30.0, help="RX front-end gain [dB]")
    ap.add_argument("--ant", default="")
    ap.add_argument("--clock-source", default="")
    ap.add_argument("--key", default="SESSION-KEY")
    ap.add_argument("--mode", default="ctr", choices=["ctr", "gcm"])
    ap.add_argument("--frames", type=int, default=1024, help="max frames to decode")
    ap.add_argument("--ckpt", default="run/lpi_v4.best.pt")
    ap.add_argument("--config", default="run/lpi_v4.best.json")
    ap.add_argument("--capture", default="", help="decode a .npz/.csv/.cs16/.c64 file instead")
    ap.add_argument("--save", default="", help="also save the live capture to .npz")
    ap.add_argument("--watch", action="store_true", help="keep acquiring + decoding")
    ap.add_argument("--json-out", default="", help="append one JSON line per capture")
    ap.add_argument("--expect", default="",
                    help="fail (exit 1) unless this exact text is recovered -- makes "
                         "the receiver usable as a CI/nightly bench test")
    ap.add_argument("--uhd-info", action="store_true")
    ap.add_argument("--coarse", default="auto",
                    help='"auto" = search for the LO offset only if the direct lock '
                         'fails; "off" = never; a number = always search +/- that many Hz')
    ap.add_argument("--coarse-span", type=float, default=6000.0,
                    help="coarse CFO search half-span [Hz]; 2 free-running B210 LOs at "
                         "2.484 GHz are within ~5 kHz of each other")
    ap.add_argument("--selftest", action="store_true",
                    help="no radio: TX->RX loopback through a synthetic channel")
    args = ap.parse_args()

    if args.uhd_info:
        from uhd_io import find_devices
        print(find_devices())
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    cfg.fs = args.rate if not args.capture else cfg.fs
    dec = load_rx(cfg, args.ckpt, device)
    np.set_printoptions(linewidth=160)

    if args.selftest:
        from lpi_core import apply_channel, LPIGenerator
        from lpi_crypto import FrameCodec as FC
        from tx_usrp import build_iq
        torch.manual_seed(0)
        gen = LPIGenerator(cfg).to(device)
        fcx = FC(args.key, cfg.n_bits, mode=args.mode)
        msg = "SELFTEST-SHADOWCOMM-v4-LPI-0042"
        bits = fcx.encode_long(msg)
        iq = build_iq(cfg, gen, bits, args.key, device, seed=3)
        for snr in (10.0, 5.0, 0.0):
            raw, _ = apply_channel(torch.from_numpy(
                np.stack([iq.real, iq.imag])[None].astype(np.float32)), cfg,
                snr_db=snr, impairments=True)
            z = (raw[0, 0].numpy() + 1j * raw[0, 1].numpy())
            t0 = time.perf_counter()
            res = decode_capture(z, cfg, dec, args.key, args.mode, args.frames, device,
                           coarse="auto")
            res["need"] = fcx.frames
            res["searched"] = f"{z.size} samples"
            print(f"[selftest] SNR {snr:5.1f} dB:")
            print_result(res, time.perf_counter() - t0)
            if res.get("text") == msg and res.get("ok"):
                print(f"           text matches exactly -> link OK")
        print("[selftest] if the 10 dB line decodes, the sync+decode path is fine and "
              "any field problem is RF (freq/rate/gain/clock).")
        return 0

    if args.capture:
        if args.capture.endswith(".npz"):
            d = np.load(args.capture, allow_pickle=True)
            raw = d["iq"] if "iq" in d else d["data"]
            if "fs" in d:
                cfg.fs = float(d["fs"])
            meta = {k: str(d[k]) for k in d.files}
            print(f"[rx] capture {args.capture}: {raw.size} samples @ {cfg.fs:g} S/s "
                  f"= {raw.size/cfg.fs:.3f} s")
        elif args.capture.endswith((".cs16", ".ic16")):          # GNU Radio int16
            a = np.fromfile(args.capture, dtype=np.int16).reshape(-1, 2)
            raw = (a[:, 0] + 1j * a[:, 1]).astype(np.float32) / 32768.0
            print(f"[rx] {args.capture}: {raw.size} interleaved-int16 samples "
                  f"(assumed fs={cfg.fs:g}; override with --rate)")
        elif args.capture.endswith((".c64", ".fc32", ".complex", ".dat")):
            raw = np.fromfile(args.capture, dtype=np.complex64)
            print(f"[rx] {args.capture}: {raw.size} complex64 samples "
                  f"(assumed fs={cfg.fs:g}; override with --rate)")
        else:
            raw = np.loadtxt(args.capture, delimiter=",", dtype=np.complex64)
            print(f"[rx] csv capture: {raw.size} samples")
        t0 = time.perf_counter()
        res = decode_capture(raw, cfg, dec, args.key, args.mode, args.frames, device,
                           coarse=args.coarse, span=args.coarse_span)
        res["need"] = FrameCodec(args.key, cfg.n_bits, mode=args.mode).frames
        res["searched"] = f"{raw.size} samples (file)"
        print_result(res, time.perf_counter() - t0)
        if args.expect and args.expect not in res.get("text", ""):
            print(f"[rx] EXPECTED {args.expect!r} -> exit 1")
            return 1
        return 0

    from uhd_io import Radio
    rad = Radio(addr=args.addr, rate=args.rate, freq=args.freq, gain=args.gain,
                ant=args.ant, clock=args.clock_source, direction="rx")
    cfg.fs = rad.rate
    print(f"[rx] {rad.summary()} | key={args.key!r} mode={args.mode}")
    n = max(1, int(round(args.seconds * rad.rate)))
    fcx = FrameCodec(args.key, cfg.n_bits, mode=args.mode)
    try:
        while True:
            t0 = time.perf_counter()
            raw = rad.recv(n)
            res = decode_capture(raw, cfg, dec, args.key, args.mode, args.frames, device,
                           coarse=args.coarse, span=args.coarse_span)
            res["need"] = fcx.frames
            res["rate"] = rad.rate
            res["freq"] = args.freq
            res["gain"] = rad.gain
            res["seconds"] = n / rad.rate
            print_result(res, time.perf_counter() - t0)
            if args.expect and args.expect not in res.get("text", ""):
                print(f"[rx] EXPECTED {args.expect!r} -> exit 1")
                return 1
            if args.save:
                np.savez_compressed(args.save, iq=raw, fs=rad.rate, freq=args.freq,
                                    gain=rad.gain, key=args.key)
                print(f"           saved {args.save} ({raw.size} samples)")
            if args.json_out:
                with open(args.json_out, "a") as f:
                    f.write(json.dumps({k: v for k, v in res.items()
                                        if isinstance(v, (int, float, str, bool, type(None)))}) + "\n")
            if not args.watch:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[rx] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
