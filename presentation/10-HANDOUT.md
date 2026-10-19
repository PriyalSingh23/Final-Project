# PROJECT ANSHUMAN — one-page summary

**AI-generated LPI waveform: AES-encrypted text over a USRP B210 that is
statistically indistinguishable from thermal noise.**

ShadowComm LPI-CGAN v4 · MCTE Mhow + IIT Indore · IEEE-GCON 2026 accepted ·
Elsevier journal draft in preparation

---

## The problem in one line

Encryption protects **content**. It does nothing about **existence** — an
adversary does not need your words, only the fact that you transmitted and the
grid square you transmitted from.

Steps 1–3 of the intercept kill chain (energy detection, direction finding,
traffic analysis) need nothing from your cryptography. Step 4 — AI
classification — reaches 100 % accuracy on BPSK with as few as **ten** training
samples.

## What we built

A conditional GAN learns a **keyed spreading codebook**. 128 message bits are
spread over 4 chips each into a 512-sample frame; the sum of 128 independent
keyed chips is Gaussian by the central limit theorem, which is where most of
the stealth comes from for free. A small learned refinement head removes the
residual structure that a trained detector could exploit.

```
TX  plaintext → AES-128-CTR (fresh nonce) → CRC-8 → RS(42,34) → 336 bits
    → CGAN generator (z₆₄ + 128 bits → 2×512) → keyed Zadoff–Chu pilot
    → USRP B210 @ 2.484 GHz, 245760 S/s

RX  B210 → keyed sync → CFO/phase correction → matched filter + learned head
    → RS decode → AES decrypt → plaintext (exact match or nothing)
```

Frame 512 + 64 pilot samples = **2.3438 ms**. One burst = 3 frames = **7.03 ms**
= 24 characters. Processing gain **9.03 dB**. Gross rate **54.6 kb/s**.

## Measured results

`python lpi_v4/lpi_eval.py --ckpt run/lpi_v4.best.pt --frames 512 --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400` → **exit code 0, 6/6 gate checks PASS**

| | target | measured |
|---|---|---|
| KS p-value vs AWGN | > 0.05, want > 0.85 | **0.8646** |
| EW domains passed | all 8 | **8/8**, composite Sc 0.140 |
| Adversary CNN, held out | 48–56 % | **55.9 %** (AUC 0.586) at 400 fit steps · 58.8 % at 800 |
| Field BER, 5 real impairments | < 1 % | **3.05e-05** @ 6 dB, 0.0 @ 8–10 dB |
| Message FER | < 5 % | **0 %** at every SNR ≥ 2 dB, 64/64 exact (bound 4.6 %) |

Same benchmark, classical waveforms — every one separated from noise at 100 %:
BPSK KS p 0.145 · QPSK 0.000 · DSSS 0.002 · FHSS 0.001 · OFDM 0.507.
Ours: 0.870. Few-shot: ours stays at 37.5–52.1 % up to n = 2000; BPSK is at
100 % from n = 10.

## What is *not* proved

Live UHD streaming at 245760 S/s on hardware (scripts written, offline
equivalents pass) · PA-on operation · range and multipath beyond bench-top
line of sight · the USRP's leaked transmit-LO line, which our own receiver uses
for CFO estimation and an eavesdropper can therefore see · key management
beyond a passphrase · anti-replay · cryptographic certification.

## Reproduce it, no radio required

```bash
python -m pytest tests -q -m "not slow"     # 20 passed
python lpi_v4/check_docs.py                 # 82 documented commands verified
python lpi_v4/lpi_grc.py                    # the exact blocks the flowgraphs embed
python lpi_v4/rx_usrp.py --selftest         # TX → 5-impairment channel → RX → text
python lpi_v4/lpi_eval.py --ckpt lpi_v4/run/lpi_v4.best.pt --frames 512 \
       --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400     # the gate, ~4 min
```

## Where it would be used

Patrol and section-level comms in an EM-contested zone · EMCON discipline
(24 characters without being located) · UAS datalinks that do not give away the
ground-control station · counter-UAS RF deception, where few-shot resistance is
the operative property.

Form factors: bench (2× B210 + 2 laptops) · man-portable (Pi 5 + B210,
~15–20 W) · vehicle/rack (X310-class + GPSDO).

**As a trials and training asset, yes. As an operational cryptographic radio,
not until key management, propagation trials and certification are complete.**

---

*Repository: `github.com/PriyalSingh23/Final-Project` · `FINDINGS.md` records
every failed experiment and the measurement that found it ·
`FIELD_TEST_CHECKLIST.md` is the one-page bench procedure.*
