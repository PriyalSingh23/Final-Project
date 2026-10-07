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

What does *not* work, measured: raising the adversarial pressure. A 30-epoch
run at `--w-adv 1.5 --d-stop-frac 0.55 --eps 0.10` (vs the shipped model's
`--w-adv 1.0 --eps 0.10`) behaved like this:

| epochs | `adv` | KS p | `Sc` |
|---|---|---|---|
| 5–12 | 50–55 % | 0.80 → 0.999 | 0.127–0.138 |
| 13–24 (wardens still training) | 57 → 74 % | 0.02–0.47 | 0.16–0.29 |
| 26–34 (wardens frozen) | 65 → 72 % | 0.00–0.31 | 0.19–0.29 |

i.e. once the generator is optimised hard against *one particular* discriminator,
it distorts the samples to confuse that net, and the eight-domain battery
punishes it for it. The statistical losses, not the adversarial weight, are what
hold Gaussianity; the adversarial term only needs to be enough to stop the
warden from *winning*. (Same reason the trainer now refuses to pick a "best"
checkpoint from epochs where the warden was never trained.)

The GAN's *own* discriminator is a different animal: it co-adapts with the
generator, so `adv` in `run/*.csv` swings with the epoch (50 % at epoch 3, 84 %
by epoch 41 in one run). v4 therefore freezes the wardens after `--d-stop-frac`
of the run (the paper's protocol: train the detector, then measure it) and only
accepts a "best checkpoint" from epochs inside that frozen regime.

**The adversary number belongs to the detector, not only to the generator.** Re-fitting
the held-out CNN with 800 steps instead of 400 on the *same frozen generator* moved the
score from 55.90 % (AUC 0.5858) to **58.80 %** (AUC 0.6164) — the 48–56 % band is met at
the paper's budget and missed at twice it, which is what "the generator is only as
invisible as the detector you tried" means in practice. Both numbers are in
`run/eval.json`-style output now (`fit_steps` is recorded next to `fit_adv_acc`), so a
reviewer can see which claim came from which budget instead of picking the flattering one.

### Trying to buy the adversary band with `loss_cov` (a measured dead end)

`loss_cov` was the one term that ever moved `adv`, so the obvious experiment was to
double it: `--epochs 45 --steps 24 --batch 48 --eps 0.10 --w-cov 3.0` at the shipped
seed (only the weight differs), 45 epochs, best snapshot at epoch 36. Decision rule was
fixed *before* measuring: promote only if the adversary lands in 48--56 % at **both**
400 and 800 fitting steps while KS p and `Sc` stay at or better than the shipped model.

| | shipped (`--w-cov 2.0`) | `--w-cov 3.0` |
|---|---|---|
| KS p | 0.8646 | **0.0131** |
| composite `Sc` | 0.1399 | 0.3263 |
| domains passed | 8/8 | 7/8 (circularity 0.0107 → 0.0130·10 = 0.1303) |
| adversary, held-out | 55.90 % / 58.80 % | 56.70 % / 58.20 % |
| field BER @ 6 dB | 3.05e-5 | 6.10e-5 |

The rule says no, and the shape of the failure is the lesson: more weight on one
statistical term traded away the headline metric for ~1 point of adversary margin, and
did not even fix the 800-step reading. It joins `--w-adv 1.5` as a dead end, written here
so the next person does not spend a training budget on it. Nothing about the worse
checkpoint is committed -- `run/*` scratch is gitignored -- but the row above is
reproducible with the command quoted, and the shipped `run/lpi_v4.best.pt` stands.

**A rate of 1e-5 is two bit errors, so quote the bound, not the point.** The same
`--w-cov 3.0` checkpoint measured field BER at 6 dB as 6.10e-5 with `--fit-warden 400`
and 3.05e-5 with `--fit-warden 800` -- the fitting pass consumes global RNG, so the
channel realisation differs, i.e. one or two flipped bits out of 65,536. `lpi_eval.py`
now reports the one-sided 95 % Clopper--Pearson bound per in-band SNR (`field_ci95`,
`fer_ci95`) and refuses to let a claim it cannot support read as a pass.

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
* **A CLI flag that was used but never defined** (`lpi_eval.py` still wrote its
  report under `if args.md:` while the `--md` definition had been dropped by an
  earlier edit). Nothing in the test suite touches it, `--help` does not mention
  it, and the failure only appears when a caller passes the flag — which was CI.
  Two different exit codes from one cause: `--md x` → argparse
  `unrecognized arguments` (exit 2), no `--md` → `AttributeError` (exit 1). Both
  read as "the metric gate failed" in the Actions UI, so an afternoon went into
  the wrong conclusion. Guarded now by
  `tests/test_lpi_v4.py::test_cli_flags_are_all_defined`, which AST-diffs every
  `args.<name>` against the `add_argument` dests of every entry point; it fails
  when the flag is removed, which is how it was verified.
* In a GitHub Actions step, relative paths resolve against `working-directory`,
  not the repo root: `open("lpi_v4/run/ci_eval.json")` inside a step already
  `cd`-ed into `lpi_v4` is a `FileNotFoundError` that looks like a bad metric.
* `export_for_grc()` wrote `generator_lpi.pt`/`decoder_lpi.pt` **next to whatever
  checkpoint was trained**, so a 6-epoch smoke test silently replaced the exports
  the `.grc` flowgraphs and `tx_usrp.py` load (this was caught by `git status`
  showing those two files modified after a CI rehearsal). It is now `--export-dir`,
  and both workflows point it at a scratch folder; a run that means to feed the
  radio still exports into `run/` by default.

* **The exported TorchScript was not the measured model.** `run/generator_lpi.pt` and
  `run/decoder_lpi.pt` -- the pair a hand-written GNU Radio block loads, and the pair the
  notebook zips up as "for GNU Radio / the USRP scripts" -- disagreed with
  `run/lpi_v4.best.pt` in **28 of 28** decoder tensors (max |Δ| 0.198). It was not the
  legacy v1 export either (that file is 512 kB, this one 648 kB); which epoch of which
  run produced it is no longer recoverable, because the tuning run dirs that held the
  other snapshots were scratch and are gone. `lpi_train.py` now re-exports the *best*
  snapshot as the last thing it does, `--export-every` writes to
  `<export-dir>/intermediate/` so a mid-run peek cannot overwrite the canonical pair,
  and every export drops `export_manifest.json` with the source filename and the
  checkpoint's sha256. `tests/test_lpi_v4.py::test_exported_torchscript_matches_the_checkpoint`
  checks the numbers *and* the digest, so a mirror that is not a mirror fails the build.
  (No published metric changed: `lpi_eval.py`, `rx_usrp.py` and the v4 flowgraphs all load
  `run/lpi_v4.best.pt` itself -- the stale pair could only have bitten a custom block.)
* **A gate that measures the training budget, not the code, is worse than no gate.**
  `ci.yml` used to assert `ks>0.05` on its 6-epoch/12-step smoke model; it went green on
  one commit and red on the next with no functional change (the trainer gained one
  `torch.randn`-consuming export call, which shifted which epoch became "best"). CI now
  asks only whether the receive chain works (`ber_field<1%` + `capture_decodes`) and
  prints the stealth numbers as artefacts; stealth regression is the nightly's job, where
  the model is actually converged. If a threshold is straddled at the budget you can
  afford to test with, report it, do not gate on it.
* **A traced module can bake in the example input's shape.** `export_for_grc` traces at
  batch 2 (`torch.jit.trace` warns that a `size_prods == 1` comparison became a constant,
  which is exactly this risk) while the GRC blocks call the decoder at batch 1. Measured: the
  exported decoder is bitwise identical to the eager one at batch 1, 2, 8 and 32 (max
  |Δ| = 0.0), and the generator returns unit-power `(B, 2, 512)` at all four -- no batch
  assumption leaked in. `test_exported_torchscript_matches_the_checkpoint` covers batch 1 for
  that reason; `check_trace=False` on its own cannot tell you.

* **Documentation is an interface, so it gets a test.** `lpi_v4/check_docs.py` parses every
  command line in the READMEs, the field checklist, `v4.bat`, the workflows, the notebook
  *and* the scripts' own docstrings/`print` hints, resolves each named script, and checks
  each `--flag` against that script's argparse (77 references right now). It found the
  `--md` story above, a `link_test.py --full` hint pointing at a file nobody wrote,
  and `lpi_grc.py --selftest` documented but undefined (the self-test *was* the default
  behaviour, so the flag is now real). Same reasoning as the paper's thresholds: a number
  nobody checks drifts.

## 8. Current state of every acceptance metric

One command produced this column, on the shipped checkpoint, on CPU:
`lpi_eval.py --ckpt run/lpi_v4.best.pt --frames 512 --snrs 0,2,4,6,8,10 --n-msg 64
--fit-warden 400` (gate band `--gate-snr-min 5` dB). `--n-msg 64` is not decoration: at
16 texts per SNR a clean in-band point only bounds FER at 8.9 %, i.e. the run could not
have supported the `< 5 %` line however well it went; 64 with none lost bounds it at
4.6 %, which does. Quote it together with the
numbers: KS p and FER move with `--frames`/`--n-msg`, and the adversary moves with
`--fit-warden`, so a number without its protocol is not reproducible.

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
| adversary accuracy | 48–56 % | 55.90 % (400 fit steps), AUC 0.5858 — **58.80 %** / AUC 0.6164 at 800 steps | PASS at the paper's detector budget, *missed* at 2× it (§4) |
| BER, cable | < 1 % | 0.0 at 8/10 dB, 3.05e-5 at 6 dB, 2.14e-3 at 2 dB; 1.00e-2 at 0 dB | PASS for every SNR ≥ 2 dB (0 dB is below the claimed operating point) |
| BER, antenna | < 5 % | field path (5 impairments) 2.44e-4 at 4 dB, 0.0 at ≥8 dB | PASS (to be confirmed on hardware) |
| message FER (CRC+RS+AES) | < 5 % | 0 % at every SNR ≥ 2 dB (64 texts each, all exact); 95 % upper bound **4.6 %**; 18.8 % at 0 dB | PASS, and the sample supports it |
| cyclostationary ratio | ≈ 1× | 1.013 | PASS |

Not yet demonstrated (needs the bench, not the sandbox): live UHD streaming at
245760 S/s with `tx_usrp.py`/`rx_usrp.py`, and the PA-on / antenna-separation
sweep. The flowgraphs and both scripts are written for it and their offline
equivalents pass (`rx_usrp.py --selftest`, `lpi_grc.py`, `make_synth_capture.py`
+ `ota_verify.py` for the legacy path).
