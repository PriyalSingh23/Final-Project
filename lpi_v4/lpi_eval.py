#!/usr/bin/env python3
"""
lpi_eval.py -- the one script that decides whether ShadowComm LPI v4 is real.
===============================================================================

It runs four blocks and prints a PASS/FAIL table against the field gate:

  1. STEALTH   ew_report() + a 50/50 adversary (accuracy, AUC, calibration)
  2. LINK      per-bit BER vs SNR: matched-filter-only / in-graph (AWGN) /
               FIELD (pilot sync + CFO + phase noise + DC + IQ imbalance)
  3. MESSAGE   the thing that actually matters: text -> CRC-8 -> RS(42,34) ->
               AES-128 -> keyed spread -> IQ -> channel -> sync -> descramble ->
               text, reported as frame-error rate and recovered characters
  4. CAPTURE   the same chain on a raw .npz capture (synthetic or USRP), and --
               optionally -- a live USRP receive (--uhd ADDR)

Usage
-----
    python lpi_eval.py --ckpt run/lpi_v4.best.pt
    python lpi_eval.py --ckpt run/lpi_v4.best.pt --selftest      # plumbing only
    python lpi_eval.py --ckpt run/lpi_v4.best.pt --write-capture cap.npz
    python lpi_eval.py --ckpt run/lpi_v4.best.pt --capture cap.npz
    python lpi_eval.py --ckpt run/lpi_v4.best.pt --uhd 192.168.10.2 --md report.md

numpy/torch only -- no GNU Radio needed for blocks 1-4.
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
from lpi_core import (LPIConfig, LPIGenerator, LPIDecoder, LockNotFound, Warden,
                      apply_channel, channel_and_sync, frame_with_pilot, frames_to_bits,
                      load_config, make_preamble, receiver_front_end,
                      strip_preamble, sync_and_correct, unit_power)
from lpi_crypto import FrameCodec
from lpi_stats import ew_report

GATE = {"adv": 56.0, "ks": 0.05, "ber_field": 0.01, "fer": 0.05}


# ------------------------------------------------------------------ model ----
def load_models(ckpt, cfg, key, device, selftest=False):
    torch.manual_seed(0)
    gen = LPIGenerator(cfg).to(device)
    dec = LPIDecoder(cfg).to(device)
    dec.tie_to(gen)
    wd = Warden(input="iq", ch=cfg.ch).to(device)
    wd_trained = False
    if not selftest:
        pass
    if selftest:
        print("[eval] --selftest: untrained networks (plumbing check only)")
        return gen, dec, wd, cfg
    if not os.path.exists(ckpt):
        raise SystemExit(f"[eval] no checkpoint at {ckpt} -- run lpi_train.py first "
                         f"(or pass --selftest)")
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    if not isinstance(blob, dict):                      # bare state_dict
        blob = {"generator": blob}
    pick = lambda *names: next((blob[n] for n in names if n in blob), None)
    gsd, dsd = pick("generator", "gen", "g_state"), pick("decoder", "dec", "d_state")
    missing_g = gen.load_state_dict(gsd, strict=False).missing_keys
    if dsd is not None:
        dec.load_state_dict(dsd, strict=False)
    dec.tie_to(gen)
    csrc = pick("cfg", "config")
    if isinstance(csrc, dict):
        cfg = LPIConfig(**{k: v for k, v in csrc.items() if k in LPIConfig.__dataclass_fields__})
    if "epoch" in blob:
        print(f"[eval] checkpoint epoch {blob['epoch']}"
              + (f" metrics {blob['metrics']}" if blob.get("metrics") else ""))
    if missing_g:
        print(f"[eval] WARNING unloaded generator keys: {missing_g[:6]}")
    if "warden1" in blob:
        wd.load_state_dict(blob["warden1"], strict=False)
        wd_trained = True
    for m in (gen, dec, wd):
        m.eval()
    print(f"[eval] warden weights: {'from checkpoint' if wd_trained else 'NOT in checkpoint'}")
    return gen, dec, wd, cfg


def gen_frames(gen, cfg, n, device, seed=0):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    bits = (torch.randint(0, 2, (n, cfg.n_bits), generator=rng).float() * 2 - 1)
    z = torch.randn(n, cfg.z_dim, generator=rng)
    with torch.no_grad():
        x = gen(z.to(device), bits.to(device))
    return x, (bits.numpy() > 0).astype(np.uint8).reshape(-1)


# ----------------------------------------------------------------- blocks ----
def fit_warden(gen, cfg, device, n_train=2600, n_test=1000, steps=400, ch=32, lr=1e-4):
    """Table-5 protocol: a *fresh* CNN detector is trained on a balanced pool
    (GAN frames vs reference noise), 70/15/15, then scored on held-out frames.
    This is the honest 'adversary accuracy' number -- not the GAN's own
    discriminator, which is a moving target by construction."""
    from lpi_core import Warden as W
    tr, te = [], []
    # Disjoint pools, and that is the whole point: sharing one seed between the
    # train and test split makes the test frames *literally the training frames*,
    # and a 4-layer CNN will happily report 80 % on that.  Measured twice, once
    # with the leak, once without -- the leak is worth ~30 points.
    for split, nn_, seed_f, seed_r in ((tr, n_train, 917, 918), (te, n_test, 4571, 4572)):
        x, _ = gen_frames(gen, cfg, nn_ // 2, device, seed=seed_f)
        g = torch.Generator(device="cpu").manual_seed(seed_r)
        awgn = unit_power(torch.randn(nn_ // 2, 2, cfg.frame_len, generator=g).to(device))
        split.append((unit_power(x), awgn))
    W1 = W(n_fft=cfg.frame_len, input="iq", ch=ch).to(device)
    opt = torch.optim.Adam(W1.parameters(), lr=lr, betas=(0.5, 0.999))
    torch.set_grad_enabled(True)
    xh, yh = torch.cat([a for a, _ in tr]), torch.cat([b for _, b in tr])
    for _ in range(steps):
        i = torch.randint(0, xh.shape[0], (xh.shape[0],), device=device)
        j = torch.randint(0, xh.shape[0], (xh.shape[0],), device=device)
        f, r = xh[i[:16]], yh[j[:16]]
        logits = torch.cat([W1(f).flatten(), W1(r).flatten()])
        labels = torch.cat([torch.zeros(16, device=device), torch.ones(16, device=device)])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(W1.parameters(), 1.0)
        opt.step()
    torch.set_grad_enabled(False)
    W1.eval()
    xf, xr = torch.cat([a for a, _ in te]), torch.cat([b for _, b in te])
    p = torch.cat([torch.sigmoid(W1(xf)).flatten(), torch.sigmoid(W1(xr)).flatten()])
    y = torch.cat([torch.zeros_like(p[: xf.shape[0]]), torch.ones_like(p[: xf.shape[0]])])
    acc = float((((p > 0.5).float() == y.float()).float().mean()) * 100)
    pv = p.cpu().numpy(); yv = y.cpu().numpy() > 0.5
    order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(pv.size)
    n1, n0 = int(yv.sum()), int((~yv).sum())
    auc = float((ranks[yv].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5
    print(f"[stealth] fitted detector (Table-5 protocol, disjoint pools): "
          f"bal.acc {acc:.1f}% AUC {auc:.3f} "
          f"(train {xh.shape[0]} frames, test {xf.shape[0]}, {steps} steps)")
    W1 = W1.cpu(); W1.cpu()
    return {"adv_acc": acc, "adv_auc": auc, "n_train": int(xh.shape[0]),
            "n_test": int(xf.shape[0]), "steps": steps,
            "state": {k: v.cpu() for k, v in W1.state_dict().items()}}


def block_stealth(gen, dec, wd, cfg, device, n=4096, fit_steps=0):
    x, _ = gen_frames(gen, cfg, n, device, seed=1)
    with torch.no_grad():
        sig = unit_power(x).numpy().astype(np.float32)
        ref = torch.randn(n, 2, cfg.frame_len, generator=torch.Generator().manual_seed(2))
    rep = ew_report(sig, ref.numpy())
    # adversary
    torch.manual_seed(3)
    a = unit_power(x)
    b = ref.to(device)
    with torch.no_grad():
        pa = torch.sigmoid(wd(a)).view(-1).cpu().numpy()
        pb = torch.sigmoid(wd(b)).view(-1).cpu().numpy()
    yhat = np.concatenate([np.ones_like(pa), np.zeros_like(pb)])
    p = np.concatenate([pa, pb])
    acc = float(((p > 0.5) == (yhat > 0.5)).mean() * 100)
    acc = max(acc, 100.0 - acc)          # label inversion is free for the warden
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(p.size)
    n1, n0 = int(yhat.sum()), int((1 - yhat).sum())
    auc = float((ranks[yhat > 0.5].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5
    auc = max(auc, 1.0 - auc)
    print(f"[stealth] KS p={rep['ks_p']:.3f} kurt={rep['kurt']:.3f} H={rep['entropy']:.3f} "
          f"circ={rep['circ']:.3f} PAPRdev={rep['papr']:+.2f}dB SCF={rep['scf']:.3f} "
          f"C42={rep['c42']:+.3f} WVD={rep['wvd']:.3f} -> Sc={rep['Sc']:.3f} "
          f"({rep['n_pass']}/8)")
    print(f"[stealth] in-training warden bal.acc={acc:.1f}% AUC={auc:.3f} "
          f"mean p(fake)={p[yhat>0.5].mean():.3f} vs p(AWGN)={p[yhat<0.5].mean():.3f}")
    out = {**{k: rep[k] for k in ("ks_p", "kurt", "entropy", "circ", "papr", "scf", "c42",
                                  "wvd", "Sc", "n_pass")},
           "adv_acc": acc, "adv_auc": auc}
    if fit_steps:
        fw = fit_warden(gen, cfg, device, steps=fit_steps)
        st = fw.pop("state")
        out.update({f"fit_{k}": v for k, v in fw.items()})
        torch.save({"warden1": st, "note": "detector fitted with the Table-5 protocol"},
                   "run/warden_fitted.pt")
        print(f"[stealth] saved run/warden_fitted.pt (reusable detector)")
    return out


def block_link(gen, dec, cfg, device, snrs, n=512, key="SESSION-KEY"):
    T = cfg.frame_len
    rows = {}
    x, bits = gen_frames(gen, cfg, n, device, seed=4)
    for snr in snrs:
        # (a) matched filter only, AWGN, perfect frame grid
        y, p = apply_channel(x, cfg, snr_db=snr, impairments=False)
        mf = (torch.sign(dec.despread(unit_power(y))[0]) > 0).float()
        mf_ber = float((mf.cpu().numpy().reshape(-1) != bits).mean())
        # (b) full learned decoder, AWGN, in-graph sync
        yb, pb = channel_and_sync(x, cfg, snr_db=snr, impairments=False)
        with torch.no_grad():
            db = dec.hard_bits(yb).cpu().numpy().reshape(-1)
        g_ber = float((db != bits).mean())
        # (c) FIELD: impairments + pilot sync + front end
        pay = (x[:, 0] + 1j * x[:, 1]).cpu().numpy()
        pl = make_preamble(key, cfg.pilot_len)
        raw = np.concatenate([np.zeros(int(cfg.fs) // 200, np.complex64),
                              frame_with_pilot(pay, pl).reshape(-1),
                              np.zeros(int(cfg.fs) // 200, np.complex64)]).astype(np.complex64)
        raw, chan = apply_channel_torch(raw, cfg, snr, device)
        try:
            frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n,
                                            cfo_search=True, return_info=True)
        except LockNotFound as e:
            info = {"offset": -1, "cfo_hz": 0.0, "n_frames": 0, "pilot_snr_db": float("nan"),
                    "dropped_frames": -1, "note": str(e)[:60]}
            frames = np.zeros((0, 2, cfg.frame_len), np.float32)
        if frames.shape[0] == 0:
            rows[snr] = dict(mf=mf_ber, graph=g_ber, field=1.0, fer=1.0, **info)
            continue
        got = frames_to_bits(dec, frames[:n], device)
        f_ber = float((got != bits[: got.size]).mean())
        nf = got.size // cfg.n_bits
        ref = bits.reshape(-1, cfg.n_bits)[: nf]
        fer = float((got.reshape(nf, cfg.n_bits) != ref).any(1).mean())
        rows[snr] = dict(mf=mf_ber, graph=g_ber, field=f_ber, fer=fer, **info)
        print(f"[link] {snr:5.1f} dB | MF {mf_ber:.2e} | in-graph {g_ber:.2e} | "
              f"field {f_ber:.2e} (FER {fer:.1%}) | offset {info['offset']} "
              f"CFO {info['cfo_hz']:+.1f} Hz pilot-SNR {info['pilot_snr_db']:.1f} dB "
              f"dropped {info['dropped_frames']}")
    return rows


def apply_channel_torch(raw, cfg, snr, device):
    """Run the SAME torch channel model (CFO/phase noise/DC/IQ imbalance/AWGN) on
    a flat numpy capture, by treating it as one long (1, 2, N) signal."""
    v = np.stack([raw.real, raw.imag], axis=0)[None].astype(np.float32)
    t = torch.from_numpy(v).to(device)
    L = cfg.pilot_len + cfg.frame_len
    n = (t.shape[-1] - 1) // L + 1
    pad = n * L - t.shape[-1]
    t = torch.nn.functional.pad(t, (0, pad)).view(1, 2, n, L).flatten(2)
    y, p = apply_channel(t, cfg, snr_db=snr, impairments=True)
    y = y.view(1, 2, n, L).flatten(2)[:, :, : t.shape[-1]]
    return (y[0, 0].cpu().numpy() + 1j * y[0, 1].cpu().numpy()), p


def block_message(gen, dec, cfg, device, snrs, key, n_msg=8, mode="ctr"):
    fc = FrameCodec(key, cfg.n_bits, mode=mode)
    frames_per_burst = fc.frames
    pl = make_preamble(key, cfg.pilot_len)
    out = {}
    for snr in snrs:
        msgs = [f"LPI-v4|SNR{snr:g}|seq{i:02d}|pl-{i * 7919 % 9973:04d}"[: fc.max_len]
                for i in range(n_msg)]
        ok_txt, nbits_e, nbits_t, nfer = 0, 0, 0, 0
        bits_all, burst_ref = [], []
        for i, m in enumerate(msgs):
            b = fc.encode(m)
            rng = np.random.default_rng(1000 + i)
            bits_all.append(b)
            burst_ref.append((m, b))
        bits = np.concatenate(bits_all)
        n = bits.size // cfg.n_bits
        x = torch.from_numpy(bits.reshape(n, cfg.n_bits).astype(np.float32) * 2 - 1)
        g = torch.Generator(device="cpu").manual_seed(7)
        z = torch.randn(n, cfg.z_dim, generator=g)
        with torch.no_grad():
            pay = gen(z.to(device), x.to(device))
        payc = (pay[:, 0] + 1j * pay[:, 1]).cpu().numpy()
        raw = np.concatenate([np.zeros(int(cfg.fs) // 200, np.complex64),
                              frame_with_pilot(payc, pl).reshape(-1),
                              np.zeros(int(cfg.fs) // 100, np.complex64)]).astype(np.complex64)
        raw, p = apply_channel_torch(raw, cfg, snr, device)
        try:
            frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n,
                                            cfo_search=True, return_info=True)
        except LockNotFound as e:
            info = {"offset": -1, "cfo_hz": 0.0, "n_frames": 0,
                    "pilot_snr_db": float("nan"), "dropped_frames": -1,
                    "note": str(e)[:60]}
            frames = np.zeros((0, 2, T), np.float32)
        got = frames_to_bits(dec, frames, device) if frames.shape[0] else np.zeros(0, np.uint8)
        nrec = got.size // cfg.n_bits
        nbits_t = int(n) * cfg.n_bits
        flat = bits.reshape(-1)
        nbits_e = int((got[: nrec * cfg.n_bits] != flat[: nrec * cfg.n_bits]).sum())
        for j, (m, ref) in enumerate(burst_ref):
            s = slice(j * frames_per_burst * cfg.n_bits,
                      (j + 1) * frames_per_burst * cfg.n_bits)
            chunk = got[s.start:s.stop]
            if chunk.size < s.stop - s.start:
                nfer += 1
                continue
            back, good = fc.decode(chunk)
            nfer += int(back != m or not good)
            ok_txt += int(back == m and good)
        fer = nfer / len(msgs)
        out[snr] = dict(fer=fer, ber=(nbits_e / nbits_t if nbits_t else 1.0),
                        msgs_ok=ok_txt, n=len(msgs), **info)
        print(f"[msg] {snr:5.1f} dB | BER {nbits_t and nbits_e/nbits_t:.2e} | FER {fer:6.1%} "
              f"| exact messages {ok_txt}/{len(msgs)} | dropped {info['dropped_frames']}")
    return out


def block_capture(gen, dec, cfg, device, args, snr, key="SESSION-KEY"):
    pl = make_preamble(key, cfg.pilot_len)
    n = args.frames
    if args.capture:
        d = np.load(args.capture, allow_pickle=True)
        raw = d["iq"] if "iq" in d else d["data"]
        bits = d["bits"] if "bits" in d else None
        src = args.capture
    else:
        x, bits = gen_frames(gen, cfg, n, device, seed=11)
        pay = (x[:, 0] + 1j * x[:, 1]).cpu().numpy()
        raw = np.concatenate([np.zeros(int(cfg.fs) // 200, np.complex64),
                              frame_with_pilot(pay, pl).reshape(-1),
                              np.zeros(int(cfg.fs) // 200, np.complex64)]).astype(np.complex64)
        raw, _ = apply_channel_torch(raw, cfg, snr, device)
        src = "synthetic"
        if args.write_capture:
            np.savez_compressed(args.write_capture, iq=raw, bits=bits, fs=cfg.fs,
                                key=cfg.key, snr_db=snr, frame_len=cfg.frame_len,
                                pilot_len=cfg.pilot_len, n_bits=cfg.n_bits)
            print(f"[capture] wrote {args.write_capture}  ({raw.size} samples, "
                  f"{raw.size/cfg.fs*1e3:.1f} ms)")
    t0 = time.perf_counter()
    try:
        frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=n,
                                        cfo_search=True, return_info=True)
    except LockNotFound as e:
        frames, info = np.zeros((0, 2, cfg.frame_len), np.float32), {
            "offset": -1, "cfo_hz": 0.0, "n_frames": 0, "pilot_snr_db": float("nan"),
            "dropped_frames": -1, "note": str(e)[:60]}
    dt = time.perf_counter() - t0
    if frames.shape[0] == 0:
        print(f"[capture] {src}: NO FRAME FOUND in {raw.size} samples "
              f"({raw.size/cfg.fs:.3f} s) -- check fs/gain/timing")
        return {"found": 0, "ber": 1.0, "sync_s": dt}
    got = frames_to_bits(dec, frames, device)
    ber = 1.0
    if bits is not None:
        b = np.asarray(bits).reshape(-1).astype(np.uint8)[: got.size]
        ber = float((got[: b.size] != b).mean())
    print(f"[capture] {src}: {frames.shape[0]} frames in {dt:.2f} s | offset "
          f"{info['offset']} CFO {info['cfo_hz']:+.1f} Hz BER {ber:.2e}")
    return {"found": int(frames.shape[0]), "ber": ber, "sync_s": dt, **info}


def block_usrp(gen, dec, cfg, args, key="SESSION-KEY"):
    """Live capture from a USRP, decoded through the same path as everything else."""
    from uhd_io import Radio
    rad = Radio(addr=args.uhd, rate=args.rate, freq=args.freq, gain=args.gain,
                direction="rx")
    cfg.fs = rad.rate
    n = max(1, int(round(args.seconds * rad.rate)))
    print(f"[usrp] {rad.summary()} | capturing {args.seconds:.2f} s ({n} samples)")
    t0 = time.perf_counter()
    raw = rad.recv(n)
    print(f"[usrp] {raw.size} samples in {time.perf_counter()-t0:.2f} s")
    if args.save_capture:
        np.savez_compressed(args.save_capture, iq=raw, fs=rad.rate, freq=args.freq,
                            gain=rad.gain, key=key)
        print(f"[usrp] wrote {args.save_capture}")
    from rx_usrp import decode_capture
    res = decode_capture(raw, cfg, dec, key, args.mode, args.frames,
                         next(dec.parameters()).device.type if hasattr(dec, "parameters")
                         else "cpu")
    if not res.get("n_frames"):
        print(f"[usrp] NO LOCK -- {res.get('note','')} (try --coarse, more --gain)")
        return {"found": 0, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
    print(f"[usrp] {res['n_frames']} frames | CFO {res['cfo_hz']:+.1f} Hz (coarse "
          f"{res.get('coarse_hz',0.0):+.1f}) | pilot {res['pilot_snr_db']:.1f} dB | "
          f"CRC {res.get('crc_ok')} | text {res.get('text','')[:60]!r}")
    return {"found": int(res["n_frames"]), "ber": res.get("ber", float("nan")),
            "crc_ok": res.get("ok", False), "text": res.get("text", ""),
            "cfo_hz": res["cfo_hz"], "coarse_hz": res.get("coarse_hz", 0.0),
            "pilot_snr_db": res["pilot_snr_db"]}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run/lpi_v4.best.pt")
    ap.add_argument("--config", default="")
    ap.add_argument("--key", default="SESSION-KEY")
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--snrs", default="-2,0,2,4,6,8,10")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--capture", default="")
    ap.add_argument("--write-capture", default="")
    ap.add_argument("--uhd", default="", help="USRP addr, e.g. 192.168.10.2")
    ap.add_argument("--save-capture", default="", help="with --uhd: also write this .npz")
    ap.add_argument("--rate", type=float, default=245760.0)
    ap.add_argument("--freq", type=float, default=2.484e9)
    ap.add_argument("--gain", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--mode", default="ctr", choices=["ctr", "gcm"])
    ap.add_argument("--fit-warden", type=int, default=0,
                    help="train a fresh Table-5 detector for N steps and report its "
                         "held-out balanced accuracy (the honest adversary number)")
    ap.add_argument("--n-stealth", type=int, default=4096)
    ap.add_argument("--cfo-max", type=float, default=150.0,
                    help="CFO half-range used by the FIELD block (the pilot fine sync "
                         "is unambiguous to +/-(fs/2/frame_period); larger offsets are "
                         "handled by rx_usrp.py --coarse, not by this benchmark)")
    ap.add_argument("--out-json", default="run/eval.json")
    ap.add_argument("--md", default="",
                    help="also write a human-readable Markdown report here")
    ap.add_argument("--no-fail", action="store_true",
                    help="always exit 0: print the gate but never fail the caller "
                         "(use in CI when a short run is not expected to pass it, and "
                         "in Colab so a failing metric does not kill the cell)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    gen, dec, wd, cfg = load_models(args.ckpt, cfg, args.key, device, args.selftest)
    print(f"[eval] cfg: T={cfg.frame_len} n_bits={cfg.n_bits} pilot={cfg.pilot_len} "
          f"fs={cfg.fs:g} gain={cfg.gain_db:.1f} dB | device={device}")

    if args.cfo_max > 0:
        print(f"[eval] channel CFO half-range set to +-{args.cfo_max:g} Hz "
              f"(checkpoint said +-{cfg.cfo_max:g})")
        cfg.cfo_max = float(args.cfo_max)
    snrs = [float(s) for s in args.snrs.split(",")]
    rep = {"block_stealth": block_stealth(gen, dec, wd, cfg, device,
                                          fit_steps=args.fit_warden),
           "block_link": block_link(gen, dec, cfg, device, snrs, n=args.frames, key=args.key),
           "block_message": block_message(gen, dec, cfg, device, snrs, args.key,
                                          n_msg=4, mode=args.mode),
           "block_capture": block_capture(gen, dec, cfg, device, args, snrs[len(snrs) // 2],
                                    args.key)}
    if args.uhd:
        rep["block_usrp"] = block_usrp(gen, dec, cfg, args, args.key)

    st, ln, ms = rep["block_stealth"], rep["block_link"], rep["block_message"]
    if "fit_adv_acc" in st:
        st["adv_acc"], st["adv_auc"] = st["fit_adv_acc"], st["fit_adv_auc"]
    worst = max(ln.values(), key=lambda r: r["field"])
    fer = max(r["fer"] for r in ms.values())
    checks = {
                "adv_in_50s": abs((st.get("fit_adv_acc") or st["adv_acc"]) - 50) <= 6,
        "ks>0.05": st["ks_p"] > GATE["ks"],
        "ew>=7/8": st["n_pass"] >= 7,
        "ber_field<1%": worst["field"] < GATE["ber_field"],
        "fer<5%": fer < GATE["fer"],
        "capture_decodes": rep["block_capture"].get("found", 0) > 0
        and rep["block_capture"].get("ber", 1) < GATE["ber_field"],
    }
    print("\n================= FIELD GATE =================")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  worst-field:", f"{worst['field']:.2e} dB-BER @{max(ln)}",
          "| worst-FER", f"{fer:.1%}")
    print("=====================================\n")
    rep["gate"] = checks
    rep["all_pass"] = bool(all(checks.values()))
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"[eval] wrote {args.out_json}")
    if args.md:
        with open(args.md, "w") as f:
            f.write(md_report(rep, cfg))
        print(f"[eval] wrote {args.md}")
    if args.no_fail:
        print("[eval] --no-fail: exit 0 regardless of the gate")
        return 0
    return 0 if rep["all_pass"] else 1


def md_report(rep, cfg):
    st, ln, ms = rep["block_stealth"], rep["block_link"], rep["block_message"]
    if "fit_adv_acc" in st:
        st["adv_acc"], st["adv_auc"] = st["fit_adv_acc"], st["fit_adv_auc"]
    L = ["## ShadowComm LPI v4 -- evaluation report", "",
         f"config: T={cfg.frame_len}, {cfg.n_bits} bits/frame, pilot {cfg.pilot_len}, "
         f"fs={cfg.fs:g}, processing gain {cfg.gain_db:.1f} dB", "",
         "### Stealth", "| metric | value |", "|---|---|"]
    for k in ("ks_p", "kurt", "entropy", "circ", "papr", "scf", "c42", "wvd", "Sc",
              "adv_acc", "adv_auc"):
        L.append(f"| {k} | {st[k]:.4g} |")
    L += ["", "### Link (per-bit BER)", "| SNR | MF | in-graph | field | FER |",
          "|---|---|---|---|---|"]
    for s, r in ln.items():
        L.append(f"| {s:g} | {r['mf']:.2e} | {r['graph']:.2e} | {r['field']:.2e} | "
                 f"{r['fer']:.1%} |")
    L += ["", "### Message layer (CRC-8 + RS(42,34) + AES)", "| SNR | BER | FER | msgs ok |",
          "|---|---|---|---|"]
    for s, r in ms.items():
        L.append(f"| {s:g} | {r['ber']:.2e} | {r['fer']:.1%} | {r['msgs_ok']}/{r['n']} |")
    L += ["", "### Gate", ""]
    for k, v in rep["gate"].items():
        L.append(f"- [{'x' if v else ' '}] {k}")
    L.append("")
    L.append(f"**overall: {'ALL TARGETS MET' if rep['all_pass'] else 'NOT YET PASSING'}**")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
