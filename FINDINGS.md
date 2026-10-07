# ShadowComm LPI — findings, failures and the v4 rewrite

Engineering record for `lpi_v4/`. Written so that the *negative* results are
reusable: every item below was measured, and each "do not do this" cost at least
one full training run to learn.

Numbers quoted are reproducible with `python lpi_eval.py --ckpt run/lpi_v4.best.pt
--fit-warden 800 --frames 512 --snrs 0,2,4,6,8,10` (CPU, ~4 min).

---

## 1. Why v1–v3 could not reach the metrics

| # | root cause | symptom in v1–v3 | what v4 does |
|---|---|---|---|
| 1 | **No spreading gain.** 256 bits carried on 4 real dims/bit with a dense generator: the receiver had nothing to integrate over. | BER floor 2.69 % regardless of SNR, no matter how long training ran | 128 info bits per 512-sample frame = 4 chips/bit = **9.03 dB** processing gain; a tied matched-filter is the first stage of the decoder, so the network only *refines* an already-working receiver |
| 2 | **MSE against ±1 labels.** A squared-error loss on a bipolar target has its optimum *between* the two symbols. | decoder soft outputs clustered at 0, BER stuck near 0.1 | BCE on logits (`binary_cross_entropy_with_logits`), bits = sign(logit) |
| 3 | **0.30σ i.i.d. dither with no whitening.** The masking noise is independent per sample, so a plain matched filter just sees extra noise. | stealth metrics perfect (KS p 0.99), BER 8 % | the decoder runs **ZF / decorrelating detection** (ridge-inverse of the code Gram, α = 0.05) plus the learned head; ε is now a single explicit knob (`--eps`) traded against BER |
| 4 | **Noiseless training channel.** The net never saw CFO, DC, IQ imbalance or phase noise, so the field receiver had nothing to stand on. | bench BER 0.2 %, over-the-air 0.5 (i.e. random) | five-impairment channel in-graph (`apply_channel`), trained at ±50 % of the paper's nominal values, *evaluated at nominal* |
| 5 | **Scrambled phase inside the generator.** Per-channel angles derived from `z` are not a rotation, and even a true rotation of the payload alone cannot be undone by a blind square loop (±90° ambiguity). | BER pinned at 0.39–0.50 forever while every stealth metric looked great | absolute phase is treated as a **channel** quantity: TX applies pilot+payload together, the receiver removes it per frame from that frame's own pilot. Removed from the generator entirely. |
| 6 | **Conjugate-mirrored Frank preamble.** A mirrored sequence has *two* identical correlation peaks → a 32-sample timing ambiguity → 50 % BER with no visible symptom. | "the model is broken" for a week | **keyed Zadoff-Chu** preamble (odd root derived from the session key): measured aperiodic sidelobes 0.154, single unambiguous peak, wrong key cannot lock |
| 7 | **Per-frame exact unit power on the real class only** in the warden's training set → an instantly learnable fingerprint. | adversary 59.9 % while all eight statistical domains passed | both classes are normalised identically everywhere, and the reported adversary number comes from a *held-out, disjoint-pool* detector (see §4) |

## 2. The channel is where field bugs live

* `apply_channel` defines the noise so that `snr_db` **is** the complex-baseband
  SNR: each real dimension gets variance `1/10^(snr/10)`, because the signal is
  unit power *per real dimension*. Before this, every "AWGN-only" BER in the
  project was effectively measured 3 dB optimistic.
* I/Q imbalance and CFO must be applied as **real 2×2 mixes**. An earlier
  revision wrote `z.unsqueeze(-1) * exp(1j*ang)` with `ang` of shape `(B,1,T)`,
  which broadcast to `(B,T,T)` and produced BER = 0.5 with a *healthy-looking*
  loss curve. `bce` going from 1.04 → 0.022 was that one line.
* **Ordering matters:** derotate the CFO ramp *before* measuring the pilot phase.
  The other way double-counts `2π·cfo·t_pilot`: sync looks perfect, BER is 0.5.
* The frame-grid **phase** (`offset = argmax` of the matched-filter envelope over
  `mod total_len`) and the **first active frame** are two different quantities.
  Reporting only the phase, then starting the decode at the first grid point
  *inside the leading zero pad*, shifted the whole message — that single bug
  turned `link_ber` into 0.498 at every SNR and was invisible for a day because
  the per-SNR BER was flat (a "saturated" model looks the same).
* `info["dropped_frames"]` and `info["pilot_snr_db"]` now exist precisely so the
  two failure modes above are distinguishable in one printed line.

## 3. Carrier-offset range, and the acquisition step nobody writes down

The pilot repeats once per frame, so a per-frame pilot phase only resolves CFO
**modulo** `fs/T_frame`, i.e. unambiguous to

```
± fs / (2 · T_frame) = ± 245760 / (2 · 576) = ± 213 Hz
```

Two free-running B210 LOs at 2.484 GHz are ~2 ppm apart ≈ ±5 kHz. So:

* `LPIConfig.cfo_max` was reduced 300 → **150 Hz**: the *fine* path's design
  point must sit inside its own unambiguous range, otherwise the benchmark
  measures a configuration the receiver cannot support.
* `coarse_acquisition()` (in `lpi_core.py`) is a two-pass search: ±`span` at
  `fs/(4·L_pilot)` ≈ 960 Hz scored by the preamble peak/median ratio (no
  decoder needed), then ±one bin at `fs/(16·T_payload)` ≈ 30 Hz scored by the
  decoder's mean |logit| — i.e. by *whether the bits became confident*. Verified
  end-to-end: a ±2 kHz offset capture decodes to the exact text
  (`python lpi_grc.py --cfo-max 2000`).
* `rx_usrp.py` / the GRC block call it **automatically** whenever a burst does not
  come back CRC-clean, so on the bench "wrong LO offset" is not a failure mode.

## 4. Adversary accuracy: the number, and the trap in measuring it

Two conventions matter more than any hyper-parameter:

1. **A warden may swap its own output labels for free.** Reporting "3 % accuracy"
   for a detector that is 97 % accurate *inverted* flatters the generator
   enormously. All reported `adv` numbers are `max(acc, 100 − acc)`, and AUC is
   folded to `max(AUC, 1 − AUC)`.
2. **The two classes must be disjoint pools.** The first version of
   `lpi_eval --fit-warden` seeded the training pool and the test pool with the
   same generator seed, so the test frames *were* the training frames; the CNN
   reported 80–82 % / AUC 0.90 on that. After the fix (independent pools,
   1300 train / 500 test frames, 400–2000 steps, Adam 1e-4, β=(0.5,0.999)):

   | model | fitted detector, held out | AUC | paper's Table 5 |
   |---|---|---|---|
   | v4 canonical (`--eps 0.10 --w-cov 0.5`) | **55.9 %** | 0.586 | 55.6 ± 1.2 %, AUC 0.583 |
   | v6 (`--eps 0.16 --w-cov 3.0`) | 55.4 % | 0.585 | — |

   That is the acceptance-band number (48–56 %) and it is now a *generalisation*
   number, not a memorisation one. `run/warden_fitted.pt` is saved so the exact
   detector can be re-scored by a reviewer.

The GAN's *own* discriminator is a different animal: it co-adapts with the
generator, so `adv` in `run/*.csv` swings with the epoch (50 % at epoch 3, 84 %
by epoch 41 in one run). v4 therefore freezes the wardens after `--d-stop-frac`
of the run (the paper's protocol: train the detector, then measure it) and only
accepts a "best checkpoint" from epochs inside that frozen regime.

## 5. What the statistical losses are actually for

Measured with a **completely untrained** generator (random weights, 512 frames):

* `ew_report` → `ks_p 0.857, kurt 2.998, entropy 0.959, circ 0.041, SCF 1.016,
  C42 −0.002, WVD 1.000, 8/8 pass`
* payload BER through the *field* path (keyed sync, CFO, phase noise, DC, IQ
  imbalance): **1.6e-4**.

Conclusion: the Gaussian/stealth side comes from the **central limit theorem over
128 keyed chips per frame** plus the ε-blend — it is free, and needs no tuning.
What the losses buy is (a) keeping it there while the waveform is made
*invertible*, and (b) the covariance-domain term (`loss_cov`, §6). Do not "tune
for KS": tune for the two things that fight each other (ε and `--w-rec`) and let
the CLT do the rest.

## 6. `loss_cov`: matching the domain the detector looks in

Any generator with an internal dimension ≪ the sample count produces frames that
live in a fixed low-rank subspace, so the batch Gram spectrum has outliers where
i.i.d. Gaussian frames follow Marchenko–Pastur. That is precisely the statistic
the *eigenvalue detector* family (the standard tool in cooperative spectrum
sensing) uses, and it is not visible in any of the eight per-frame domains.
`lpi_stats.loss_cov` penalises the log-spectrum mismatch between the fake and
reference batches (48×48 Gram, `eigvalsh`, cheap). Reference values: 3e-4 for
AWGN-vs-AWGN, 0.285 for a rank-128 signal-vs-AWGN. Weight: 0.5 in the canonical
run, 3.0 in v6.

It is a *detection-domain* loss, so it is also the honest answer to "why does
your adversary stay at 55 %": the generator is optimised against the class of
detectors that could tell it apart, not merely against a scalar KS test.

## 7. Bugs worth remembering (all silent, all fixed)

* `np.correlate(a, conj(a))` conjugates its **second** argument: that is
  `Σ a[n+d]a[n]`, not the autocorrelation. The peak of a Zadoff-Chu looked
  *absent*. Write `np.correlate(a, a)`.
* `np.packbits/unpackbits` default to **MSB-first** (`bitorder="little"` is the
  other choice), and GRC's `blocks.unpack_k_bits_bb(k=8)` is MSB-first too. Two
  conventions that agree by accident will disagree the moment someone "makes it
  explicit", so `bitorder="big"` is spelled out at every call site and
  `tests/test_lpi_v4.py::test_codec_bit_order_matches_numpy` pins the
  pack/unpack round-trip that `make_bits_file.py` relies on.
* `torch.stft` on a complex input raises "Casting complex values to real";
  the real STFT for the Warden's TF map is built with `unfold`+`fft` instead.
* `RSCodec(nsym)` defaults to `nsize=255`: a 42-byte block does **not**
  round-trip. Use `RSCodec(8, nsize=42)`. Symptom: `decode()` returned
  `('', ok=True)` because an all-zero codeword has a valid CRC-8 of 0 and n=0
  was being swallowed by the "empty message" branch.
* A receive burst must be read with a stride of `frames·n_bits` (**384**), not
  `CODEWORD_BITS` (336): the payload is zero-padded to a whole number of frames.
* AES-GCM with a truncated tag needs `mac_len` at *creation* time, or
  `decrypt_and_verify` always fails.
* `nohup … &` inside some shells returns immediately but dies with the session —
  long runs need a real process manager (see `AUTOMATION.md`).

## 8. Current state of every acceptance metric

| gate (checklist) | target | v4 canonical | verdict |
|---|---|---|---|
| KS p (IQ vs AWGN) | > 0.05, want > 0.85 | 0.865 | **PASS**, in the "want" band |
| kurtosis | ≈ 3.0 | 2.991 | PASS |
| entropy H | > 0.95 | 0.957 | PASS |
| circularity | < 0.15 | 0.011 | PASS |
| PAPR deviation | < 3 dB | +0.01 dB | PASS |
| SCF ratio | < 3 | 1.013 | PASS |
| C42 | < 0.3 | −0.005 | PASS |
| WVD TF ratio | < 3.7 | 1.017 | PASS |
| composite `Sc` | < 0.20 (Good) | 0.140 | PASS ("Excellent" is < 0.10) |
| adversary accuracy | 48–56 % | 55.9 % (held-out), AUC 0.586 | PASS |
| BER, cable | < 1 % | 0.0 at 5 dB, 1.9e-3 at 2 dB | PASS |
| BER, antenna | < 5 % | field path 0.0 at 5 dB incl. all 5 impairments | PASS (to be confirmed on hardware) |
| message FER (CRC+RS+AES) | — | 0 % at 5 dB and 2 dB, 4/4 exact texts | PASS |
| cyclostationary ratio | ≈ 1× | 1.013 | PASS |

Not yet demonstrated (needs the bench, not the sandbox): live UHD streaming at
245760 S/s with `tx_usrp.py`/`rx_usrp.py`, and the PA-on / antenna-separation
sweep. The flowgraphs and both scripts are written for it and their offline
equivalents pass (`rx_usrp.py --selftest`, `lpi_grc.py`, `make_synth_capture.py`
+ `ota_verify.py` for the legacy path).
