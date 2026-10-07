# lpi_v4 — the working ShadowComm LPI stack

Everything needed to (a) train the covert waveform, (b) prove it meets the metric
targets, and (c) put it on a USRP. Five modules, no GNU Radio dependency, no GPU
dependency:

```
lpi_core.py     config + generator + decoder + warden + channel + frame sync
lpi_stats.py    the 8-domain EW battery (Table 4 of the paper) + composite loss
lpi_crypto.py   payload layer: CRC-8 -> RS(42,34) -> AES-128-CTR/GCM -> bits
lpi_datasets.py GOLD / RadioML2016 loader for "hard negative" real captures
lpi_train.py    the trainer (composite objective, per-epoch gate, auto-export)
lpi_eval.py     the gate: stealth + per-SNR BER + message FER + capture decode
tx_usrp.py      real-time transmit         rx_usrp.py   real-time receive + decode
uhd_io.py       version-tolerant UHD wrapper (send/recv, sc16 packing)
lpi_grc.py      the two GNU Radio blocks (pure python, testable without GR)
make_grc.py     regenerates tx_lpi_v4.grc / rx_lpi_v4.grc from lpi_grc.py
make_bits_file.py  writes the coded .dat that the GRC transmitter reads
check_docs.py   proves the commands in these docs still match argparse
v4.bat          Windows entry point for all of the above
run/            checkpoints, per-epoch CSV, eval reports, exported TorchScript
                (+ export_manifest.json: which checkpoint that TorchScript came from)
```

**One source of truth for the physics.** `LPIConfig` defines frame length, pilot
length, bits/frame, sample rate and ε; `sync_and_correct` is *the* receiver
synchroniser used by the trainer's link metric, the evaluator, the USRP scripts
and the GRC block. That is deliberate: the v1–v3 "512 vs 1024 sample frame"
disaster was possible only because two files each had their own idea.

## 5 minutes, no radio

```bash
python -m pytest ../tests -q -m "not slow"      # framing + sync + crypto loopback
python lpi_train.py --epochs 12 --steps 24 --batch 48 --out run/smoke.pt \
    --export-dir run/smoke_export
python lpi_eval.py --ckpt run/smoke.best.pt --frames 128 --snrs 2,5,8 --no-fail
python rx_usrp.py --selftest                    # TX -> channel -> RX -> text
python lpi_grc.py                               # the GRC blocks, without GRC
```

`--export-dir` matters for a throwaway run: without it the TorchScript exports land
in `run/`, which is the folder the `.grc` flowgraphs and `tx_usrp.py` read the
*shipped* model from. `--no-fail` prints the gate but always exits 0, which is what
you want from a 12-epoch model (and from a Colab cell, where a non-zero exit kills
the rest of the notebook). `lpi_eval.py` exits non-zero only when you leave it off —
that is the mode CI uses once the model is expected to pass.

## Train for the paper numbers

```bash
python lpi_train.py --epochs 45 --steps 24 --batch 48 \
    --eps 0.10 --out run/lpi_v4.pt --log run/lpi_v4.csv
# GPU / Colab:  --epochs 200 --steps 60 --batch 96
# with real captures as hard negatives:  --radioml --data GOLD_XYZ_OSC.0001_1024.hdf5
python lpi_eval.py --ckpt run/lpi_v4.best.pt --frames 512 --snrs 0,2,4,6,8,10 \
    --n-msg 64 --fit-warden 400 --md run/eval.md --out-json run/eval.json \
    --latex run/eval.tex
```

`--latex` writes the paper's two metric tables from the run itself (balanced
environments, no packages assumed, `yes`/`no` rather than `\checkmark`), with the command
line that produced them in a comment -- which is the only way the manuscript and the
measurement stay in agreement. `lpi_v4/lpi_latex.py` is standalone and tested against a
saved `--out-json`, so you can also re-render tables from an old run.

The epoch row printed by the trainer *is* the gate: `bce` (decodability), `link`
(BER through the field synchroniser), `adv` (warden balanced accuracy), `KS p /
kurt / H / circ` (noise-likeness), `EW n/8` (how many of the eight domains pass),
`Sc` (the composite distance to AWGN).

Knobs, in the order they matter: `--eps` (masking fraction — the single
stealth↔BER trade), `--w-rec`, `--w-cov` (the covariance/Gram term),
`--frame-len`/`--n-bits` (processing gain), `--d-stop-frac` (when the wardens get
frozen). `--w-adv` is deliberately left at 1.0: a measured 30-epoch run with
`--w-adv 1.5 --d-stop-frac 0.55` drove KS p from 0.87 down to 0.10–0.30 and left
the warden at 65–72 % — over-fitting *one* discriminator warps the samples without
buying stealth (see `../FINDINGS.md` §4). If the adversary is your failing metric,
add statistical pressure (`--w-stat`, `--w-cov`) before adversarial pressure. Do not add a payload phase scramble, do not blend dither
against an un-normalised signal, and do not use a conjugate-mirrored preamble —
see `../FINDINGS.md` §1.

## Measured (CPU, this repo, `run/lpi_v4.best.pt`)

Every number below is from one command, run on the shipped checkpoint:
`lpi_eval.py --frames 512 --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400`
(gate band `--gate-snr-min 5`, i.e. the SNR the acceptance line is claimed at).

| | target | measured |
|---|---|---|
| KS p (IQ vs AWGN) | > 0.05 (want > 0.85) | **0.8646** |
| kurtosis / entropy / circularity | 3.0 / >0.95 / <0.15 | 2.9907 / 0.9566 / 0.0107 |
| PAPR dev / SCF / C42 / WVD | <3 dB / <3 / <0.3 / <3.7 | +0.009 dB / 1.0129 / −0.0048 / 1.0166 |
| composite `Sc` | <0.20 | **0.1399** (8/8 domains pass) |
| adversary, held-out fitted detector | 48–56 % | **55.90 %**, AUC 0.5858 |
| field BER (sync + CFO + phase noise + DC + IQ) | <1 % | 3.05e-5 @6 dB · **0.0** @8,10 dB · 2.14e-3 @2 dB · 1.00e-2 @0 dB |
| message FER (CRC-8 + RS + AES), 64 texts/SNR | <5 % in band | **0 %** at every SNR ≥ 2 dB; 18.8 % @0 dB |
| 95 % upper bounds (worst in-band point) | ≤ target | **BER ≤ 9.6e-5** (65,536 bits), **FER ≤ 4.6 %** (64 texts) |
| `lpi_eval.py` exit code | 0 | **0** (6/6 gate checks PASS) |

The bound row is the honest version of the two rows above it. A point estimate of 0 %
only says what the sample excluded, so `lpi_eval` computes the one-sided
Clopper--Pearson upper bound for every in-band point and marks a check `n/a` when the
bound does not clear the target. The sequence, all measured on this checkpoint with a
clean in-band run: 4 texts -> bound 52.7 % (proves nothing), 16 -> 17.1 %, 32 -> 8.9 %
(still short of 5 %), **64 -> 4.6 %**, which is why `--n-msg 64` is the recommended
command and not a stylistic choice. `cp_n_needed()` in `lpi_eval.py` is that arithmetic
in reverse -- 59 texts for the FER line, 299 bits for the BER line -- and the JSON
carries both as `texts_needed` / `bits_needed` next to the bounds, so the sample size a
claim needs travels with the claim.

Two things that are easy to get wrong when quoting this table, both measured:

* **The adversary number is a property of the detector's budget.** The same frozen
  generator reads 55.90 % (AUC 0.586) with 400 fitting steps and **58.80 %**
  (AUC 0.616) with 800 -- i.e. the 48–56 % target is met at the paper's budget and
  missed at 2× it. Report the budget with the accuracy, or the number means nothing.
* **FER resolution is 1/messages.** With the old `n_msg=4` the metric could only move
  in 25 % steps, so "0 %" was barely a measurement; at 16 msgs/SNR it still comes out
  0 % above 2 dB, which is now worth saying.

`--gate-snr-min` (default 5 dB) is why the BER/FER rows read "worst in the band":
the sweep reaches 1.00e-2 at 0 dB, and a gate that takes the worst over *whatever*
SNR list you typed is measuring your command line, not the link. The full sweep is
printed under the gate line and kept in the JSON either way.

Reproduce with the two commands above; the definitions of each number (and the
two conventions that change them by 25 points if you get them wrong — label
inversion, disjoint pools) are in `../FINDINGS.md` §4.

## USRP

```bash
# TX box
python tx_usrp.py --addr 192.168.10.2 --freq 2.484e9 --rate 245760 --gain 0 \
                  --text "ALPHA-INDIA-001" --bursts 20 --loop --period 0.2
# RX box
python rx_usrp.py --addr 192.168.10.1 --freq 2.484e9 --rate 245760 --gain 30 \
                  --watch --json-out rx_log.jsonl
# no radio / no network: write it, read it
python tx_usrp.py --file tx.npz --text "hello" && python rx_usrp.py --capture tx.npz
```

Rate: a B210 cannot go below ~208 kS/s, so **32 kS/s in the old checklist is not
reachable** — the stack defaults to 245760 S/s (61.44 MHz / 250, exact integer
decimation, no fractional clock error). The LPI margin comes from the waveform
(9.03 dB processing gain, 128 bits per 2.34 ms frame), not from a slow sample
rate. A frame is `512 payload + 64 pilot` samples; one AES/RS burst is 3 frames.

If the receiver prints `NO LOCK`, it is one of exactly four things — frequency,
sample rate, gain, key — and the printed `pilot-SNR` / `CFO` / `dropped` fields
tell you which. A residual LO offset up to ±6 kHz is found automatically by the
two-pass `coarse_acquisition` (see `../FINDINGS.md` §3); `--coarse 20000` widens
it, `--coarse off` disables it.

## GNU Radio

Open `tx_lpi_v4.grc` / `rx_lpi_v4.grc` in GRC. Each embeds one small Python block
that only delegates to `lpi_grc.py`; the `lpi_dir` variable must point at this
folder. Before touching the GUI, run `python lpi_grc.py` — it drives the same two
blocks through a synthetic channel and prints the recovered text, so a broken
flowgraph is visibly *not* a broken PHY. After editing the embedded sources,
regenerate the files: `python make_grc.py`.

Payload bits for the GUI transmitter come from the crypto layer, not from the
flowgraph: `python make_bits_file.py --text "..." --bursts 8 --out /tmp/lpi_bits.dat`.
