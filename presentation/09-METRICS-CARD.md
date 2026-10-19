# Metrics card — where every number comes from

Three provenance classes. Know which class a number is in before you say it.

- **[R]** Re-measured in this workspace on 2026-10-18. Strongest: you can say
  "I ran this myself today".
- **[V4]** Recorded in `lpi_v4/run/` from the shipped checkpoint `lpi_v4.best.pt`
  (epoch 16), and consistent with [R].
- **[P]** From the two reference papers (IEEE-GCON 2026 accepted; Elsevier
  journal draft). Say "the paper reports…" for these.

---

## [R] Re-measured today

Command and exact results:

```
$ python -m pytest tests -q -m "not slow"
20 passed, 2 deselected, 4 warnings in 2.42s

$ python lpi_v4/check_docs.py
[check_docs] OK: 82 documented commands reference real flags

$ python lpi_v4/lpi_grc.py
[grc-selftest] TxCore/RxCore loopback over the modelled channel (SNR 8.0 dB,
  24 bursts, ckpt=lpi_v4/run/lpi_v4.best.pt)
[grc-selftest] frames=72 cfo=-31.5 Hz pilot=24.4 dB
  text='SHADOWCOMM-v4-GRC-LOOPBASHADOWCOMM-v4-GR' -> PASS

$ python lpi_v4/rx_usrp.py --selftest
[rx] weights: run/lpi_v4.best.pt (epoch 16)
SNR 10.0 dB: frames 6 | offset 0 | CFO +39.5 Hz (coarse +0.0) |
  pilot-SNR 31.9 dB | dropped 0 | bursts 2 crc 2/2 OK
  -> 'SELFTEST-SHADOWCOMM-v4-LPI-0042' | text matches exactly -> link OK
SNR  5.0 dB: frames 6 | CFO +137.8 Hz | pilot-SNR 28.1 dB | bursts 2 crc 0/2 FAIL
SNR  0.0 dB: NO LOCK in 3456 samples (peak/median 4.40 < 4.5)

$ python lpi_v4/lpi_eval.py --ckpt run/lpi_v4.best.pt --frames 512 \
      --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400 \
      --md run/eval_arena.md --out-json run/eval_arena.json
================= FIELD GATE =================
  [PASS] adv_in_50s      [PASS] ks>0.05       [PASS] ew>=7/8
  [PASS] ber_field<1%    [PASS] fer<5%        [PASS] capture_decodes
  worst-field: BER 3.05e-05 @6 dB | worst-FER 0.0% over 64 msgs/SNR
  95 % upper bounds: BER <= 9.6e-05, FER <= 4.6%
=====================================
```

**Note the two honest lines in the selftest output.** The 5 dB line fails CRC
and the 0 dB line does not lock. The 10 dB line proves the sync+decode path,
and the tool itself says so. If you quote the selftest, quote all three lines.

**And quote the per-capture number with it.** A single short capture decodes
**68–72 %** of the time (flat in burst count and in SNR from 6 to 10 dB),
because the CFO slope is least-squares-fitted over only `n_frames` pilot points
and can land in the wrong de-alias branch. With the shipped ARQ repeat the
console measured **40/40 exact round-trips at 6 dB and 40/40 at 10 dB** over
its own HTTP API, using 1–4 attempts (mean 2.4–2.6). "0 % FER" from
`lpi_eval.py` is a 64-message, one-sync protocol number — always say which of
the two you are quoting.

---

## [V4] The v4 results table — one command, one checkpoint

Source: `lpi_v4/run/eval.md`, `eval.json`, `metrics/last.json` (dated
2026-10-07), re-confirmed by the [R] run above.

| Metric | Value | Target |
|---|---|---|
| KS p (IQ vs AWGN) | 0.8646 | > 0.05, want > 0.85 |
| kurtosis | 2.9907 | ≈ 3.0 |
| spectral entropy | 0.9566 | > 0.95 |
| IQ circularity | 0.0107 | < 0.15 |
| PAPR deviation | +0.0087 dB | < 3 dB |
| SCF ratio | 1.0129 | < 3 |
| HOS C42 | −0.0048 | < 0.3 |
| WVD TF ratio | 1.0166 | < 3.7 |
| composite Sc | 0.1399, 8/8 | < 0.20 |
| adversary accuracy | 55.90 %, AUC 0.5858 (400 fit steps) | 48–56 % |
| adversary, 800 fit steps | 58.80 %, AUC 0.6164 | — (report it) |

Per-SNR field BER / FER (512 frames, 65,536 bits and 64 texts per point):

| SNR | field BER | BER 95 % ≤ | FER | FER 95 % ≤ | msgs |
|---|---|---|---|---|---|
| 0 dB | 1.00e-2 | 1.07e-2 | 66.4 % | 69.9 % | 52/64 |
| 2 dB | 2.14e-3 | 2.46e-3 | 0.0 % | 4.6 % | 64/64 |
| 4 dB | 2.44e-4 | 3.71e-4 | 0.0 % | 4.6 % | 64/64 |
| 6 dB | 3.05e-5 | 9.61e-5 | 0.0 % | 1.2 % | 64/64 |
| 8 dB | 0.0 | 4.57e-5 | 0.0 % | 0.6 % | 64/64 |
| 10 dB | 0.0 | 4.57e-5 | 0.0 % | 0.6 % | 64/64 |

Config (`lpi_v4/run/lpi_config.json`): n_bits 128, frame_len 512, pilot_len 64,
z_dim 64, dither_eps 0.10, refine_gain 0.20, ch 32, fs 245760,
center_freq 2.484 GHz, eval_snr 5.0, phase_noise 0.05 rad, dc_offset 0.02,
iq_amp 0.5 dB, iq_phase 1.5°, **cfo_max 300** (see caveat below).

---

## [V4] The failure-story numbers (`FINDINGS.md`)

| Claim | Number |
|---|---|
| v1–v3 BER floor | 2.69 % regardless of SNR |
| MSE-against-±1 decoder | soft outputs cluster at 0, BER ≈ 0.1 |
| i.i.d. dither, no whitening | KS p 0.99 while BER 8 % |
| noiseless training channel | bench BER 0.2 %, over-the-air 0.5 |
| generator-scrambled phase | BER pinned 0.39–0.50 |
| conjugate-mirrored preamble | 50 % BER, no symptom; keyed Zadoff–Chu sidelobes 0.154 |
| warden trained on its own test frames | 80–82 %, AUC 0.90 → 55.9 % after the fix |
| `--w-adv 1.5` dead end | KS p 0.87 → 0.10–0.30, warden 65–72 % |
| `--w-cov 3.0` dead end | KS p 0.0131, Sc 0.3263, 7/8 domains |
| **untrained generator** | KS p 0.857, kurt 2.998, entropy 0.959, circ 0.041, SCF 1.016, 8/8, BER 1.6e-4 |
| exported TorchScript ≠ checkpoint | 28 of 28 decoder tensors differed, max |Δ| 0.198 |
| FER sample arithmetic | 4 texts → 52.7 %, 16 → 17.1 %, 32 → 8.9 %, 64 → 4.6 % |
| `loss_cov` references | 3e-4 AWGN-vs-AWGN, 0.285 rank-128-signal-vs-AWGN |
| dither SNR cost | 0.19 dB at the −5 dB operating point |

---

## [P] Reference papers

**Elsevier draft** — *"On Evasion of EW Detection with Generative Waveform
Synthesis"*, A. Sharma (MCTE Mhow / IIT Indore), G. Sharma (MCTE), S. Gautam
(IIT Indore).

| Item | Value |
|---|---|
| Training | 500 epochs, batch 128, Adam β=(0.5,0.999), η_G 2e-4, η_D 1e-4, T4, nine runs, 915 MHz and 2.484 GHz |
| Carrier / rate | 2.484 GHz (ISM, Wi-Fi ch. 14), 2 MSPS, 512 IQ samples/frame, −5 dB operating SNR |
| Payload | AES-128-ctr, 128-bit nonce, 336-bit RS+CRC frame, ≤ 21 characters, latent 64 |
| Eight-domain results | KS p 0.870, kurtosis 3.009, entropy 0.994, circularity 0.036, PAPR dev +0.94 dB, SCF 0.970, C42 +0.031, WVD 1.275, **Sc 0.0809 (Excellent)**, 8/8 |
| AWGN reference row | KS p 0.789, kurt 2.987, entropy 1.002, circ 0.055, SCF 1.130, WVD 1.186 |
| BPSK row | KS p 0.145, kurt 2.272, entropy 0.844, circ 0.735, PAPR dev −3.16, SCF 0.896, C42 −0.548, WVD 0.248 |
| Detector | RadioML-style CNN, 3200 balanced samples, 70/15/15 split, four Conv1D (7,5,3,3), ten seeds |
| Headline accuracy | **55.6 % ± 1.2 %**, 95 % CI [54.74, 56.46], t(9)=14.76, p<0.001, **AUC 0.583** |
| Benchmark (bal. acc.) | LPI 55.6 · AWGN 53.5 · NRPCS 100 · OFDM 100 · FHSS 100 · QPSK 100 · BPSK 100 · DSSS 100 |
| Stealth scores | LPI 0.888 · AWGN 0.930 · all others 0.000 |
| Few-shot (n / LPI / BPSK) | 10/46.3/100 · 25/52.1/100 · 50/46.2/100 · 100/47.0/100 · 200/37.5/100 · 500/50.8/100 · 1000/51.6/100 · 2000/48.1/100 |
| Simulated BER | 0.0000 at −10, −5, 0, +5, +10 dB; 500 frames/SNR = 168,000 bits; 95 % CI ≤ 1.8e-5 |
| OTA | BladeRF Micro 2.0, 2.484 GHz, TX Windows / RX Linux, gain 40/30 dB, full duplex |
| OTA capture | 1619.7 ms, 3,239,460 samples, 2296/6327 active windows (36.3 %), 96 sync peaks, 50 decoded |
| OTA result | per-window SNR mean 9.7 dB, max 21.5 dB; 50/50 windows; 16,800 bits; **BER 0.0000**; confidence 1.0; message `ALPHA_SECURE_IND_001A` |
| Loss weights | D1 1.0 · D2 0.5 · mean 0.5 · var 0.5 · power constr. 3.0 · spectral 3.0 · cyclo 2.5 · power norm 6.0 · diversity 3.0 |
| Defences | dither σ_d = 0.30 σ̂ₓ (0.19 dB cost) · diversity regularisation · D2 reinit every 25 epochs |
| Channel | AWGN −5 dB · CFO ±10 ppm (±24.84 kHz) · phase noise 1.0° rms · DC 1 % σ̂ₓ · IQ 0.5 dB / 1.5°; trained ±50 % of nominal, evaluated at nominal |

**IEEE-GCON 2026 accepted paper** — *"RF-Robust Payload-Invariant Noise-Masked
LPI Waveform Synthesis with Authenticated Encryption for SDR Links"*,
A. Sharma, S. Gautam, G. Sharma.

| Item | Value |
|---|---|
| Hardware | bladeRF, 433.5 MHz, fs = 2 MSPS, 0.3 m indoor LoS, internal reference, TX 17 dB / RX 20 dB |
| Burst | 1024-sample Zadoff–Chu pilot + 1536-sample payload (6 × 256) = 2560 samples |
| Payload | AES-GCM + RS, authenticated decode only if sync and impairments are corrected |
| KS-PSD | 0.022, held across the sweep |
| Sweep | SNR ∈ {10,15,20,25} dB × ε ∈ {0.10,0.12,0.14}; BER falls with larger ε, KS-PSD stays ≈ 0.022 |
| Wardens | PSD-CNN on Welch PSD (nperseg 1024, noverlap 512, nfft 2048, 2048-D); TF-CNN on STFT (nperseg 256, noverlap 192, nfft 512, 512×21) |
| Dataset | 1200 I/Q of length 1536, 4 balanced classes (AWGN, QPSK, OFDM, RR–PI), 720/240/240 |
| Finding | PSD-CNN separates better; TF-CNN confuses AWGN with structured modulations but keeps RR–PI distinct → report FAR/TPR under an AWGN null |
| Key caveat stated | "Low KS-PSD is necessary but not sufficient for stealth" |

---

## Caveats to state if these numbers come up

1. **"0 % FER" is a protocol number, not a per-capture number — say which one
   you are quoting.** `lpi_eval.py` concatenates 64 messages into one long
   stream and acquires sync **once**, so a single acquisition failure cannot
   appear in its FER. Measured per single short capture (1 burst = 3 frames,
   one acquisition), the decode rate is **68–72 %**, and it is essentially flat
   in both burst count (2/4/8) and SNR (6/10 dB). The cause is measured, not
   guessed: `sync_and_correct` fits the CFO slope by least squares over only
   `n_frames` pilot points, so on a 3-frame capture the estimate is noisy enough
   that the de-alias step `slope -= 426.7·round(slope/426.7)` can land in the
   wrong branch, and `timing_refine` then acquires on the wrong sample
   (BER ≈ 0.49 when that happens). **The shipped fix is ARQ:** repeat the
   codeword and re-transmit until a burst passes CRC. Measured through the web
   console's own HTTP API, 40 messages per point: **40/40 exact round-trips at
   6 dB and 40/40 at 10 dB**, needing 1–4 attempts, mean 2.4 (6 dB) and 2.6
   (10 dB). Quote it as *"99 %+ per message with ARQ over a ~70 %
   single-capture link"* — that is both true and more impressive than hiding it.

   *Two earlier explanations for this were tested and are **retracted**: it is
   not `cfo_max = 300 Hz` (all measured |CFO| ≤ 197.7 Hz, well inside the
   ±213.3 Hz unambiguous window), not the random latent seed, and not
   zero-padding (mean BER 0.144 padded vs 0.171 unpadded). If you said
   `cfo_max` out loud earlier, correct it: the config value is legal and the
   benchmark overrides it to 150 Hz anyway.*
2. **The two hardware platforms are different.** The papers validate on a
   bladeRF Micro 2.0 at 2 MSPS; the v4 stack in this repository targets a USRP
   B210 at 245760 S/s, which is what will be on the table. Do not mix the two
   sets of numbers in one sentence.
3. **The keyed preamble genuinely rejects a wrong key — verified, not assumed.**
   A wrong-root preamble produced **0/40 locks at both 6 dB and 20 dB**
   (`LockNotFound: peak/median 3.33 < 4.5`), against 15/15 locks with the
   correct key. Worth stating because it is the mechanism behind "no key, no
   signal". It was only provable after fixing a console bug in which the
   loopback re-generated the waveform with the *receiver's* key, so a wrong key
   appeared to decrypt cleanly.
