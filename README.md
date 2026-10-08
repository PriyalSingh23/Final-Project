# ShadowComm CGAN waveform-generation demo

> **Research prototype / offline simulation.** The Flask app generates IQ arrays and performs a simulated model-to-model decode. It does not control a USRP or transmit over the air. Synthetic metrics are not evidence of low probability of intercept (LPI), RF security, or channel reliability.

## Scope and implementation

The model accepts 256 bipolar message symbols and a 64-dimensional latent vector, then emits an IQ frame shaped `(2, 512)`. The decoder returns 256 bit probabilities. The current demo path is:

```text
text → AES-CTR demo wrapper → RS(255,191) encoding + byte interleaving → 256-bit frames
     → TorchScript CGAN → in-memory IQ array / .npy download
     → TorchScript decoder → Reed–Solomon decode → text
```

The app's loopback has **no RF channel** between generator and decoder. `tx_lpi_cgan.grc` and `rx_lpi_cgan.grc` are separate GNU Radio artifacts; they have not been validated by this work on hardware. The spectrum panel is simulated UI data.

The encryption path is also demonstration-only: the current implementation uses a default key if no environment key is provided, a fixed AES-CTR counter, and no authentication tag or per-message nonce. **Do not use it for sensitive messages or production security.**

## Source material and training provenance

The Google Drive folders `Colab Notebooks` and `LPI_SDR_Project` were reviewed. The Colab notebook clones this repository and contains short training experiments; the project folder contains checkpoint/log material. The supplied `datasets` folder was empty when checked, so the reported training/evaluation here uses synthetic AWGN rather than user-provided RF captures. Optional GOLD/RadioML examples, when supplied, are used only for the threat-model detector; they do not replace the generator's synthetic AWGN spectral reference.

The selected generator starts from `lpi_checkpoint_v3_610.pt` (epoch 610). The convolutional decoder plateaued above the BER screening threshold, so a global-receptive-field MLP decoder was trained with the generator frozen. Training details, selected validation point, and later non-improving runs are recorded in [`TRAINING_REPORT.md`](TRAINING_REPORT.md). The checkpoint retains the original generator; `decoder_training` records its validation result and decoder-step counter.

## Current model and offline results

The deployed pair is described by `cgan_manifest.json`; `lpi_metrics.json` is the machine-readable offline report. The app only loads report values when its checkpoint hash matches the manifest. Results below are synthetic, noiseless model diagnostics—not OTA measurements.

| Measure | Screening criterion | Latest result |
|---|---:|---:|
| Pooled marginal KS p-value vs. standard normal | > 0.05 (0.85 is aspirational) | 0.2958 — standard screen pass; aspirational target not met |
| One freshly trained CNN: generated-vs-AWGN accuracy | 48–56% | 55.2% — screen pass for this one detector |
| Generator-to-decoder bit error rate, no channel | < 1% | 1.0892% — fail |
| Short-lag autocorrelation ratio | Diagnostic only; not a formal cyclostationarity test | 2.07× |

A p-value is sample-size dependent, and one CNN is only one detector. Passing these checks does not establish covertness or RF performance. See the [RF validation plan](FIELD_TEST_CHECKLIST.md) before considering hardware testing.

## Repository layout

| Path | Purpose |
|---|---|
| `models.py` | Generator, convolutional decoder, global decoder |
| `train.py` | Joint generator/decoder training with synthetic AWGN and an auxiliary detector |
| `train_decoder_global.py` | Decoder-only training with the generator frozen |
| `finetune_decoder.py` | Fine-tune the legacy convolutional decoder |
| `sweep.py` | Resume training and log checkpoint metrics |
| `dataset_gold.py`, `dataset_radioml.py` | Optional capture loaders; datasets are not bundled |
| `test_metrics.py` | Offline marginal KS, one-CNN adversary, autocorrelation diagnostic, noiseless BER |
| `export_for_grc.py` | TorchScript export, smoke test, and hash manifest |
| `app.py`, `index.html` | Flask/Socket.IO offline demo and browser UI |
| `generator_lpi.pt`, `decoder_lpi.pt` | Deployed TorchScript models |
| `lpi_checkpoint.pt` | Selected training checkpoint |
| `tx_lpi_cgan.grc`, `rx_lpi_cgan.grc` | GNU Radio lab scaffolds; TX disabled by default; hardware behavior not verified |
| `GRC_SETUP.md` | Stepwise setup, safety gate, known sync blockers, and test sequence |
| `tests/` | Flask/API integration smoke tests |

## GNU Radio / USRP

The repository-root flowgraphs are lab scaffolds, not a ready-to-radiate link. Read [`GRC_SETUP.md`](GRC_SETUP.md) for the model block interfaces, local path configuration, TX-disabled default, hardware-specific settings, and required frame-synchronization work. Do not transmit using the placeholder frequency or gains; no OTA link has been validated.

## Install

Run commands from the repository root. Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
# Optional CPU-only PyTorch wheel (install before requirements if desired):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`h5py` is only needed for GOLD HDF5 input. To use RadioML, obtain the dataset from its provider and pass a local path; training does not need to download it automatically.

## Training and evaluation

### Train the generator/decoder

```bash
python train.py --epochs 200 --n-samples 50000 --batch 64 --out lpi_checkpoint.pt
```

To resume that checkpoint, set `--resume` and provide a total target epoch count greater than the checkpoint epoch:

```bash
python train.py --resume --epochs 620 --n-samples 50000 --batch 64 --out lpi_checkpoint.pt
```

Optional real-reference input for the detector only:

```bash
python train.py --radioml --data /path/to/captures.hdf5 --epochs 200 --out lpi_checkpoint.pt
# or use a RadioML .pkl path
python train_radioml.py --data /path/to/RML2018.01A.pkl --epochs 200 --out lpi_checkpoint.pt
```

`train_radioml.py` is a wrapper for `train.py --radioml`; despite the historical name, the optional captures are **not discriminator training data**. They are examples for the auxiliary threat-model detector. The generator's reference remains synthetic AWGN. GOLD `.h5`/`.hdf5` loading requires `h5py`.

### Continue decoder-only training

This option preserves the generator waveform distribution while adapting a full-receptive-field decoder:

```bash
python train_decoder_global.py \
  --input lpi_checkpoint_v3_610.pt \
  --output lpi_checkpoint_global_decoder.pt \
  --steps 4000 --batch 64
```

Continue from the saved output with `--resume --steps N`; each invocation starts a fresh optimizer and keeps the best validation-BER weights. It does not update the generator.

### Evaluate and export

```bash
python test_metrics.py --ckpt lpi_checkpoint.pt --json-out lpi_metrics.json
python export_for_grc.py --checkpoint lpi_checkpoint.pt
```

The evaluator uses synthetic samples and tests:

- pooled generated marginals against a standard normal using a KS test;
- one freshly trained CNN against generated samples and AWGN;
- a short autocorrelation diagnostic;
- noiseless direct generator-to-decoder BER.

The export script writes `generator_lpi.pt`, `decoder_lpi.pt`, and `cgan_manifest.json`; the manifest records SHA-256 hashes and validates the exported artifacts at batch sizes 1 and 3. Re-run evaluation and export from the **same checkpoint** so the UI can verify matching hashes.

## Run the web demo

```bash
python app.py
```

Open `http://127.0.0.1:5000`. Set `PORT` to use another port. The app expects the TorchScript files and report/manifest in the project root by default. Optional environment variables:

- `CGAN_GENERATOR`, `CGAN_DECODER` — model paths;
- `CGAN_METRICS_REPORT` — report path;
- `SHADOWCOMM_AES_KEY` — demo key override;
- `SHADOWCOMM_SECRET_KEY` — Flask session secret.

The UI can generate IQ locally, download the latest `.npy`, and run an in-memory decode. Frequency/gain values are demo settings only. The API has no authentication; keep it on a trusted machine/network and do not expose it publicly. API endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Demo UI |
| `/api/config` | GET/POST | Read/update in-memory demo settings; the AES key is never returned |
| `/api/send` | POST | Encrypt/encode a message and generate IQ; no transmission |
| `/api/last-waveform` | GET | Download the last generated IQ array (`.npy`) |
| `/api/receive` | POST | Decode the last array in a simulated model loopback |
| `/api/metrics` | GET | Hash-matched offline report, or blank metrics if none/mismatched |

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py train.py train_decoder_global.py test_metrics.py export_for_grc.py
```

## RF and legal boundary

No over-the-air test was performed for this result. Do not infer safe transmit settings, range, privacy, or LPI from the UI, model metrics, or GNU Radio file presence. Any future RF work must follow local spectrum rules, be authorized, begin with appropriately attenuated conducted testing, and use calibrated instruments and independent capture/detection analysis. The project's `ota_verify.py` is explicitly experimental and is not an LPI verdict.
