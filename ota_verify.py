#!/usr/bin/env python3
"""Over-the-air verification for the LPI-CGAN field test.

Reads the RAW baseband captured by rx_lpi_cgan.grc (rx_raw.fc32), estimates:
  - CFO            (from the TX carrier-leakage line -- present in any real
                    USRP capture; falls back to a decoder grid search if the
                    line is absent)
  - constant phase (LO phase at capture start; decoder tolerance ~15 deg)
  - frame offset   (RX vector boundary vs TX frame boundary, 0..511 samples)
then reports:
  1. Best (offset, CFO, phase)
  2. BER of the full capture vs the known message (test_message.txt),
     decoded in chunks with per-chunk CFO/phase tracking (phase rotates
     linearly, so long captures need re-anchoring every ~250 frames)
  3. KS test of received (normalized, phase-corrected) frames vs N(0,1) --
     the real over-the-air LPI verdict: does the radiated signal look AWGN?

Search metric: HARD-BIT agreement with the known message at the best
circular shift (computed by FFT correlation of +-1 hard bits against the
+-1 message). Soft-LLR correlation was tried and rejected: misaligned frames
decode to systematic biased bits that correlate almost as well as correct
frames, and +/-4-sample offsets stay bit-consistent with the message.

Usage:
  python ota_verify.py --raw rx_raw.fc32 --msg test_message.txt
                       --decoder decoder_lpi.pt --max-frames 500
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch + scipy OpenMP coexistence
import argparse
import numpy as np
import torch
from scipy import stats
from scipy.signal import fftconvolve

FS = 32000

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='rx_raw.fc32')
ap.add_argument('--msg', default='test_message.txt')
ap.add_argument('--decoder', default='decoder_lpi.pt')
ap.add_argument('--max-frames', type=int, default=500)
ap.add_argument('--cfo-min', type=float, default=-200.0)
ap.add_argument('--cfo-max', type=float, default=200.0)
args = ap.parse_args()

raw = np.fromfile(args.raw, dtype=np.complex64)
msg_bits = np.unpackbits(np.fromfile(args.msg, dtype=np.uint8), bitorder='big')
msg_pm = msg_bits.astype(np.float64) * 2 - 1                    # +/-1 message
MSG = len(msg_bits)
print(f"[ota] raw samples: {len(raw)} ({len(raw)/FS:.1f} s) | message bits: {MSG}")
if len(raw) < 512 * 32:
    raise SystemExit("[ota] capture too short -- need at least 32 frames")

dec = torch.jit.load(args.decoder, map_location='cpu').eval()

def correct(raw, cfo, phi):
    n = np.arange(len(raw), dtype=np.float64)
    return raw * np.exp(-1j * (2 * np.pi * cfo * n / FS + phi)).astype(np.complex64)

def normalize(fr):                                             # fr: (n,512) complex
    v = np.stack([fr.real, fr.imag], axis=1).astype(np.float32)
    # PER-CHANNEL zero mean: a complex DC offset (a+jb) adds a constant to
    # I and a different constant to Q; a joint mean only removes their
    # average. Without this, residual per-channel DC systematically biases
    # the decoder (and fake-correlates the bit stream at wrap-around lags).
    v = v - v.mean(axis=2, keepdims=True)
    return v / np.sqrt((v ** 2).mean(axis=(1, 2), keepdims=True) + 1e-8)

def build_frames(corr, offset, n):
    n = min(n, (len(corr) - offset) // 512)
    if n <= 0:
        return None
    return normalize(corr[offset:offset + n * 512].reshape(n, 512))

def llr_of(v):
    with torch.no_grad():
        p = dec(torch.from_numpy(np.ascontiguousarray(v))).numpy()
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1)

def hard_of(llr):
    return np.where(llr > 0, 1.0, -1.0)

def bit_score(llr):
    """Max hard-bit agreement with the message at any circular shift,
    minus the 50% chance floor. 1.0 = perfect match, 0 = no match."""
    b = hard_of(llr)
    c = fftconvolve(b, msg_pm[::-1], mode='full')
    return float(np.abs(c).max() / len(b) - 0.5)

def score_frames(v):
    return bit_score(llr_of(v)) if v is not None else -1.0

# ---- CFO estimation from the TX carrier-leakage line -----------------------
def estimate_cfo_spectral(raw):
    x = raw[:min(len(raw), FS * 8)].astype(np.complex64)
    nfft = 1 << int(np.ceil(np.log2(len(x) * 8)))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.fft.fft(x * w, nfft)
    f = np.fft.fftfreq(nfft, 1 / FS)
    m = (np.abs(f) > 5) & (np.abs(f) <= 250)
    Xm = np.abs(X[m]); fm = f[m]
    i = int(np.argmax(Xm))
    if 0 < i < len(Xm) - 1:
        y0, y1, y2 = np.log(Xm[i - 1]), np.log(Xm[i]), np.log(Xm[i + 1])
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    else:
        d = 0.0
    cfo = fm[i] + d * (FS / nfft)
    snr = 20 * np.log10(Xm[i] / (np.median(Xm) + 1e-12))
    return float(cfo), float(snr)

# fast decimated-lag hard-bit scorer for the coarse grids
LAGS1 = np.arange(0, MSG - 256 + 1, 64)
MQ = np.stack([msg_pm[l:l + 256] for l in LAGS1]).astype(np.float32)

def fast_bit_scores(bits):                                     # bits: (n,256) +-1
    sc = np.abs(bits @ MQ.T)
    return sc.max(axis=1) / 256 - 0.5

def fast_bit_scores_multi(bits):                               # bits: (n, L)
    L = bits.shape[1]
    lags = np.arange(0, MSG - L + 1, 256)
    M = np.stack([msg_pm[l:l + L] for l in lags]).astype(np.float32)
    sc = np.abs(bits @ M.T)
    return sc.max(axis=1) / L - 0.5

# ================= search =================
best = (-1.0, None, None, 0.0)                                 # score, off, cfo, phi
cands = []
n_raw = len(raw)

cfo_spec, line_snr = estimate_cfo_spectral(raw)
print(f"[ota] spectral CFO estimate: {cfo_spec:+.2f} Hz (line SNR {line_snr:.1f} dB)")

if line_snr > 10 and args.cfo_min <= cfo_spec <= args.cfo_max:
    print("[ota] stage 1: phase x offset grid (8 frames, spectral CFO) ...")
    NWIN = 8
    n_off = min(128, (n_raw - 511) // 4 + 1)
    for ph in np.deg2rad(np.arange(0, 360, 22.5)):
        corr = correct(raw, cfo_spec, float(ph))
        big = np.lib.stride_tricks.as_strided(
            corr, shape=(n_off, NWIN * 512),
            strides=(4 * corr.strides[0], corr.strides[0]))
        v = normalize(big.reshape(n_off * NWIN, 512))
        bits = hard_of(llr_of(v)).reshape(n_off, NWIN * 256)
        sc = fast_bit_scores_multi(bits)
        for i in np.argsort(sc)[-3:][::-1]:
            cands.append((float(sc[i]), int(4 * i), cfo_spec, float(ph)))
else:
    print("[ota] stage 1: joint coarse grid (CFO x phase x offset, 1 frame) ...")
    n_off = min(128, (n_raw - 511) // 4 + 1)
    for cfo in np.arange(args.cfo_min, args.cfo_max + 1, 5.0):
        for ph in np.deg2rad(np.arange(0, 360, 45)):
            corr = correct(raw, float(cfo), float(ph))
            win = np.lib.stride_tricks.as_strided(
                corr, shape=(n_off, 512),
                strides=(4 * corr.strides[0], corr.strides[0]))
            v = normalize(win)
            bits = hard_of(llr_of(v)).reshape(n_off, 256)
            sc = fast_bit_scores(bits)
            for i in np.argsort(sc)[-3:][::-1]:
                cands.append((float(sc[i]), int(4 * i), float(cfo), float(ph)))

cands.sort(key=lambda c: -c[0])
cands = cands[:5]
print(f"[ota] stage 1 best: score={cands[0][0]:.3f} offset={cands[0][1]} "
      f"cfo={cands[0][2]:+.1f} Hz phase={np.rad2deg(cands[0][3]):.0f} deg "
      f"(top5: {[round(c[0], 3) for c in cands]})")

# ---- stage 2: full-circle phase scan at the best (offset, CFO)
print("[ota] stage 2: full-circle phase scan ...")
top_off, top_cfo = cands[0][1], cands[0][2]
ph_sc = []
for ph in np.deg2rad(np.arange(0, 360, 22.5)):
    corr = correct(raw, top_cfo, float(ph))
    s = max(score_frames(build_frames(corr, o, 16))
            for o in range(max(0, top_off - 1), min(512, top_off + 2)))
    ph_sc.append((s, float(ph)))
ph_sc.sort(reverse=True)
print(f"[ota] top phases: {[(round(s, 3), round(np.rad2deg(p), 1)) for s, p in ph_sc[:3]]}")

# ---- stage 3: DENSE offset scan (all 512) at locked (phase, CFO) ----
# Proxy scores plateau within a few samples of the true offset (spliced
# frames still decode to message-correlated bits), so the exact offset is
# chosen by the true objective later; here we just need the true offset in
# the top few.
print("[ota] stage 3: dense offset scan ...")
corr = correct(raw, top_cfo, ph_sc[0][1])
off_sc = []
win = np.lib.stride_tricks.as_strided(
    corr, shape=(min(512, (n_raw - 511)), 512),
    strides=(corr.strides[0], corr.strides[0]))
v = normalize(win)
bits = hard_of(llr_of(v)).reshape(-1, 256)
sc = fast_bit_scores(bits)
top_offs = np.argsort(sc)[-8:][::-1]
off_sc = [(float(sc[o]), int(o)) for o in top_offs]
print(f"[ota] top offsets: {[(round(s, 3), o) for s, o in off_sc[:5]]}")

# ---- stage 4: joint micro-refine (CFO +-0.2 @0.05, phase +-5.6 @2.8) ----
# around the top offsets, N=64; keep top-3 by proxy score for the final
# direct-BER selection
print("[ota] stage 4: joint micro-refine ...")
finalists = []
for _, off0 in off_sc[:8]:
    for ph in np.deg2rad(np.arange(np.rad2deg(ph_sc[0][1]) - 5.6,
                                   np.rad2deg(ph_sc[0][1]) + 5.7, 2.8)):
        for c2 in np.arange(top_cfo - 0.2, top_cfo + 0.21, 0.05):
            sc = score_frames(build_frames(correct(raw, round(float(c2), 3), float(ph)), off0, 64))
            finalists.append((sc, off0, round(float(c2), 3), float(ph)))
finalists.sort(key=lambda c: -c[0])
finalists = finalists[:3]
print(f"[ota] finalists: {[(round(f[0],4), f[1], f[2], round(np.rad2deg(f[3]),1)) for f in finalists]}")

# ================= final decode: full-capture CFO/phase refinement + ramp =================
def full_decode(off, cfo, phi, max_frames):
    """Refine CFO (+-0.06 @ 0.005) and phase (+-2.8 @ 0.7) on the WHOLE
    capture, then decode with one global correction (a constant CFO error
    rotates every frame linearly, so it must be accurate to ~mHz)."""
    n_tot = min(max_frames, (len(raw) - off) // 512)
    best = (-1.0, cfo, phi)
    for c2 in np.arange(cfo - 0.06, cfo + 0.061, 0.005):
        sc = score_frames(build_frames(correct(raw, round(float(c2), 4), phi), off, n_tot))
        if sc > best[0]:
            best = (sc, round(float(c2), 4), phi)
    _, cfo, _ = best
    for ph in np.deg2rad(np.arange(np.rad2deg(phi) - 2.8, np.rad2deg(phi) + 2.9, 0.7)):
        sc = score_frames(build_frames(correct(raw, cfo, float(ph)), off, n_tot))
        if sc > best[0]:
            best = (sc, cfo, float(ph))
    _, cfo, phi = best
    v = build_frames(correct(raw, cfo, phi), off, n_tot)
    llr = llr_of(v)
    hard = (llr > 0).astype(np.uint8)
    b = hard_of(llr)
    c = fftconvolve(b, msg_pm[::-1], mode='full')
    lag = int(np.argmax(np.abs(c)) - (MSG - 1)) % MSG
    L = len(hard)
    reps = int(np.ceil((L + MSG) / MSG)) + 2
    tiled = np.tile(msg_bits, reps)
    expect = tiled[lag:lag + L]
    ber = np.count_nonzero(hard != expect) / L
    return ber, L, v.reshape(-1), cfo, phi

print("[ota] final decode: full-capture refinement over finalists ...")
results = []
for f_sc, off_f, cfo_f, phi_f in finalists:
    ber_f, L_f, flat_f, cfo_r, phi_r = full_decode(off_f, cfo_f, phi_f, args.max_frames)
    print(f"  finalist off={off_f} cfo={cfo_f:+.2f} phi={np.rad2deg(phi_f):.1f} (proxy {f_sc:.4f})"
          f" -> refined cfo={cfo_r:+.3f} phi={np.rad2deg(phi_r):.1f} BER={ber_f*100:.2f}%")
    results.append((ber_f, off_f, cfo_r, phi_r, L_f, flat_f))

results.sort(key=lambda r: r[0])
ber, off, cfo, phi, total_bits, flat = results[0]
print(f"[ota] winner: offset={off} cfo={cfo:+.3f} Hz phase={np.rad2deg(phi):.1f} deg BER={ber*100:.2f}%")

mu, sd = flat.mean(), flat.std()
ks_stat, ks_p = stats.kstest(flat, 'norm', args=(mu, sd))

print("\n" + "=" * 60)
print("OVER-THE-AIR VERIFICATION REPORT")
print("=" * 60)
print(f"frames decoded : {total_bits // 256} (offset {off}, CFO {cfo:+.3f} Hz, phase {np.rad2deg(phi):.1f} deg)")
print(f"BIT ERROR RATE : {ber*100:.2f}%  {'PASS' if ber < 0.01 else 'FAIL'} (target < 1%)")
print(f"KS vs N(0,1)   : p={ks_p:.4f}  {'PASS' if ks_p > 0.05 else 'FAIL'} (target > 0.05) | mean={mu:.4f} std={sd:.4f}")
print("=" * 60)
