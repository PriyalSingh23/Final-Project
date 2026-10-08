# Automating this project (Colab ⇄ GitHub ⇄ USRP bench)

You asked what else to use besides Antigravity + Colab. Short answer: **the
training loop, the metric gate and the radio check should all be commands that a
machine can run without you**, because the failure mode of this project is not
"the code does not run" — it is "a metric quietly stopped meaning anything"
(test leakage, an inverted accuracy convention, a bit-order mismatch, a sync
bug that looks like a saturated model). Every tool below exists to catch one of
those.

## 0. The one-command surface (already in the repo)

| task | command | notes |
|---|---|---|
| unit + loopback tests (~40 s, no GPU, no radio) | `python -m pytest tests -q -m "not slow"` | CI runs the same file on Linux **and** Windows |
| docs ↔ argparse | `python lpi_v4/check_docs.py` | 77 documented/printed commands must name real scripts and real flags; also a pytest case, so a stale hint fails CI rather than a bench run |
| train | `python lpi_v4/lpi_train.py --epochs 45 --steps 24 --batch 48` | saves every epoch; `--resume` continues; re-exports the TorchScript pair from the *best* snapshot at the end and writes `run/export_manifest.json` (sha256 of what it mirrored) |
| measure + gate (+ paper tables) | `python lpi_v4/lpi_eval.py --ckpt lpi_v4/run/lpi_v4.best.pt --fit-warden 800 --md run/eval.md --out-json run/eval.json --latex run/eval.tex` | exit code 0 = every gate passes → usable as a CI step. Any check that *could not* fail (no `--fit-warden`, no real capture) is printed as `n/a` and listed in `gate_vacuous` in the JSON, so a short run cannot be quoted as a measurement |
| bench link test (no radio) | `python lpi_v4/rx_usrp.py --selftest` | TX→channel→RX→AES/RS→text, in one process |
| GRC-path test (no GNU Radio) | `python lpi_v4/lpi_grc.py` | exercises the exact blocks the `.grc` embeds |
| radio TX / RX | `python lpi_v4/tx_usrp.py …` / `python lpi_v4/rx_usrp.py …` | raw UHD, no flowgraph needed; `--file`/`--capture` for cable-free iteration |
| Windows convenience | `lpi_v4\v4.bat train` / `eval` / `loop` / `tx` / `rx` | same commands, correct python.exe |

Keep that table true. Anything you find yourself typing by hand twice should
become either a flag on one of those five commands or a step in CI.

## 1. Google Colab (your current training box)

* **`colab/LPI_v4_Colab_Training.ipynb`** is the supported entry point — it is
  *generated* from `colab/make_notebook.py`, so its commands are the same strings
  CI runs. Regenerate with `python colab/make_notebook.py` after editing.
* Runtime → **T4**, and tick *Settings → Run in background*. Because the trainer
  writes `run/lpi_v4.pt` + the CSV every epoch, a disconnect costs at most the
  current epoch: restart with `--resume` and it picks up at the next epoch.
* Save to Drive, not to `/content`: `--out /content/drive/MyDrive/lpi/lpi_v4.pt`.
  A 45-epoch CPU run is ~9 min, a T4 run of the paper's 500 epochs is ~2 h —
  long enough that you want the file to survive the notebook.
* **Auto-commit results** (the notebook has the cell, commented out): store a
  fine-grained PAT as the Colab secret `GH_TOKEN`, then the last cells copy
  `lpi_v4.csv`, `eval.md`, `eval.json` into the repo and `git push`. Do *not*
  commit 1 MB+ `.pt` files from a sweep — only the chosen checkpoint (see §4).
* For a sweep, prefer one notebook cell that loops over `--eps` and appends one
  CSV row per run to Drive, over opening N notebooks: the gate line printed per
  epoch is already the whole comparison table.

## 2. GitHub Actions — already wired up here

* **`.github/workflows/ci.yml`** (every push/PR touching `lpi_v4/`, `tests/`,
  `colab/`): installs CPU torch, runs `pytest -m "not slow"` on
  ubuntu+windows, then a 6-epoch training run and `lpi_eval` as a **link gate**: the
  build fails if the field BER or the capture decode regress, while the stealth
  numbers (KS p, EW, adversary, FER) are printed and archived but not gated -- an
  unconverged model crosses those thresholds by luck, and a flaky gate is worse than
  none (measured: `ks>0.05` passed and failed on consecutive commits at this budget).
  Stealth belongs to the nightly, which trains 40 epochs and compares to
  `metrics/last.json`. Artefacts: the epoch CSV, the log, `ci_eval.md`. That 6-epoch run
  passes `--no-fail` — a 6-epoch model's *own* warden has barely been
  trained, so `lpi_eval`'s full-gate verdict is meaningless there; the step's
  explicit hard subset is what fails the build. Its train also uses
  `--export-dir run/ci_export`, because `run/` is where a real run puts the
  TorchScript the radio loads.
* **`.github/workflows/nightly-train.yml`** (03:17 UTC): 40 epochs at the measured
  defaults (`--eps 0.10`, default loss weights — see FINDINGS §4 for why raising
  `--w-adv` is a dead end), full `lpi_eval --fit-warden 2000 --no-fail`, compares
  against `metrics/last.json`, and opens an issue *only* on a real regression.
  Includes the trained `.pt` + that run's exported TorchScript as artefacts, so the
  nightly is a restore point.
* To make the issue-opening work, the workflow needs `issues: write` (already
  set) — no token needed. To have it *commit* `metrics/last.json` after a good
  night, add a `repository_dispatch`/push step with a PAT, or push it manually
  from Colab (§1).
* Actions minutes on a free plan are the constraint: keep `ci.yml`'s train at 6
  epochs and put real training in Colab.

## 3. Local dev (Antigravity / VS Code)

* Point the IDE interpreter at the environment that actually has torch
  (`C:\Users\yasht\radioconda\python.exe` for radio work, a plain conda env for
  training) — a `python.exe` mismatch is the usual cause of "works in the
  terminal, red in the IDE".
* **`pre-commit`** (worth adding, 2 minutes): `check-yaml`, `end-of-file-fixer`,
  `detect-private-key`, plus `python -c "import ast,sys;[ast.parse(open(f).read()) for f in sys.argv[1:]]"`
  for `*.py`. GRC YAML and `.bat` line endings are the two things that break
  silently in this repo.
* Run tests from the IDE with
  `python -m pytest tests -q -m "not slow"` and keep a launch config for
  `lpi_eval.py --ckpt run/lpi_v4.best.pt` — the *measure* step is the one people
  skip.
* For long local runs use a real process manager, not `&`:
  Linux `tmux new -s train … ` ; Windows `start "train" cmd /c lpi_v4\v4.bat train`.

## 4. Big files

* Checkpoints are ~1.2 MB each (`generator`+`decoder`+two wardens) — fine in Git,
  **but** a sweep writes 4 files per run, so ignore `lpi_v4/run/*` except the
  canonical names (already in `.gitignore`) and use **Git LFS** if you want
  history for many checkpoints:
  `git lfs install && git lfs track "*.pt" "*.npz" && git add .gitattributes`.
  Do it *before* committing the big ones; converting afterwards rewrites history.
* Captures (`.npz`, `.cs16`, `.fc32`) never go in the repo — they belong in Drive.
  The repo keeps `make_synth_capture.py` so a *reproducible* capture exists
  without uploading one.

## 5. Radio bench automation

```bash
uhd_find_devices                    # is it on the network?
uhd_fft -f 2.484e9 -s 245760 -g 30  # eyeball the spectrum / DC / IQ offset
python lpi_v4/tx_usrp.py --loop --period 0.2 --amp 0.05        # TX box
python lpi_v4/rx_usrp.py --watch --seconds 0.25 --gain 30 \
      --json-out rx_log.jsonl                                    # RX box
```

* `--json-out` gives you a *log* instead of a screen: one JSON line per capture
  with `n_frames`, `cfo_hz`, `coarse_hz`, `pilot_snr_db`, `crc_ok`, `text`.
  `python -m json.tool` / a spreadsheet on `pilot_snr_db` vs distance is the
  actual experiment for the "measured LPI range" section of the report.
* Sweep gain/distance from a script and let `rx_usrp.py`'s exit status be the
  dependent variable — add `--expect "ALPHA-INDIA"` if you want a hard pass/fail
  per capture for a for-loop.
* The `.grc` flowgraphs (`lpi_v4/tx_lpi_v4.grc`, `rx_lpi_v4.grc`) are the GUI
  route; they embed 15-line blocks that only *delegate* to `lpi_grc.py`, so the
  GUI, the scripts and the tests cannot disagree. Regenerate with
  `python lpi_v4/make_grc.py`.
* Two-machine timing: if the USRPs have a shared 10 MHz/PPS, pass
  `--clock-source external --lo-lock` on both; otherwise trust the automatic
  `--coarse` search (it resolves ±6 kHz, which is far more than 2 ppm at
  2.484 GHz).

## 6. Experiment tracking

* `--log run/x.csv` is the source of truth (one row per epoch, every loss term
  and every metric, plain text, diffable in a PR).
* `--tb run/tb` + `tensorboard --logdir run` when you want curves; the same
  scalars are in the CSV, so nothing is lost if you skip it.
* `--wandb lpi-v4` if you want the sweep UI (needs `pip install wandb` +
  `wandb login` in Colab). Do it only if you are actually going to look at the
  comparison view — the gate column in the CSV is enough for a go/no-go.
* Whatever you use, **keep `metrics/last.json` updated** (the CI compares
  against it) and keep the gate exit code as the single yes/no.

## 7. What NOT to add

* Another model file: `lpi_v4/lpi_core.py` is deliberately the only place where a
  dimension exists (frame length, pilot, n_bits, fs, ε). The v1–v3 512-vs-1024
  mismatch was possible precisely because two files each defined the frame.
* A second trainer for RadioML: `lpi_datasets.py`/`RealBank` inside
  `lpi_train.py --radioml --data <hdf5>` already is that.
* Reinforcement/WGAN variants: the critic-runaway failure mode (a `+138`
  discriminator loss drowning every statistical term) is already documented in
  the repo history; the bounded `(σ(D(x)) − 0.5)²` objective with a deadband is
  there on purpose.
