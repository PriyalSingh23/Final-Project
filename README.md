# ShadowComm LPI-CGAN: Complete Deployment Guide

> **Low Probability of Intercept (LPI) Secure Messaging Platform**
> AES-128-CTR + Reed-Solomon + Conditional GAN Covert Waveform + USRP Over-the-Air

---

## ⚡ START HERE: the v4 stack (`lpi_v4/`) is the working one

The legacy root scripts (`models.py`, `train.py`, `test_metrics.py`, `sweep.py`)
implement the v1–v3 waveform that sits on a ~2.7 % BER floor. `lpi_v4/` is the
rewrite that reaches every target, and it is self-contained: five modules, the
USRP scripts, the GNU Radio blocks, the evaluator and the tests.

```bat
lpi_v44.bat test                      :: loopback + framing tests, no GPU/radio
lpi_v44.bat train  --epochs 45        :: ~9 min on 2 CPU cores, ~25 min on a T4
lpi_v44.bat eval                       :: the field-gate report (exit 0 == all PASS)
lpi_v44.bat loop                       :: TX -> channel -> RX -> AES/RS -> text
lpi_v44.bat tx --text "ALPHA-INDIA-001" --bursts 20
lpi_v44.bat rx --watch --gain 30
```

| metric (gate) | target | v4 measured |
|---|---|---|
| KS p, IQ vs AWGN | > 0.05 (want > 0.85) | **0.865** |
| 8-domain EW battery | all pass | **8/8**, composite Sc **0.140** |
| adversary accuracy | 48–56 % | **55.9 %** held-out, AUC 0.586 |
| BER, cable | < 1 % | **0.0** at 5 dB (1.9e-3 at 2 dB) |
| message FER (CRC-8 + RS(42,34) + AES) | — | **0 %** at 5 dB and at 2 dB |
| cyclostationary / SCF ratio | ≈ 1× | 1.013 |
| kurtosis / entropy / circularity | 3.0 / >0.95 / <0.15 | 2.991 / 0.957 / 0.011 |

Read next: **`lpi_v4/README.md`** (commands, knobs, USRP + GRC), **`FINDINGS.md`**
(why v1–v3 could not reach these numbers, and the six silent bugs that had to die
first), **`AUTOMATION.md`** (Colab ⇄ Actions ⇄ bench automation),
**`colab/LPI_v4_Colab_Training.ipynb`** (paste-and-run training in Colab).

Two corrections to the older text below: a B210 cannot sample at 32 kS/s (its
floor is ≈208 kS/s, so v4 uses 245760 S/s), and the framing code is RS(42,34)
with a `nsize=42` codec — RS(255,223) does not fit a 42-byte codeword.

---

## Table of Contents

1. [What You Are Building](#1-what-you-are-building)
2. [Folder Structure](#2-folder-structure)
3. [Phase 0: Install Everything](#3-phase-0-install-everything)
4. [Phase 1: Train the CGAN](#4-phase-1-train-the-cgan)
5. [Phase 2: Export for GRC](#5-phase-2-export-for-grc)
6. [Phase 3: Verify Metrics](#6-phase-3-verify-metrics)
7. [Phase 4: GNU Radio TX/RX](#7-phase-4-gnu-radio-txrx)
8. [Phase 5: Web App Interface](#8-phase-5-web-app-interface)
9. [Phase 6: Field Deployment](#9-phase-6-field-deployment)
10. [Troubleshooting](#10-troubleshooting)
11. [Complete File Index](#11-complete-file-index)

---

## 1. What You Are Building

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRANSMITTER (TX)                                     │
│  Plaintext → AES-128-CTR → RS(255,223) → BPSK Mapper (-1,+1)               │
│  → CGAN Generator [z(64), b(256)] → Covert Waveform (2×512 IQ)              │
│  → RRC Filter → GLFSR Spreading → USRP Sink → OVER THE AIR                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECEIVER (RX)                                        │
│  USRP Source → CGAN Decoder → Recovered BPSK (-1,+1)                       │
│  → Pack K Bits → RS Decoder → AES Decrypt → Plaintext                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **AES-128-CTR**: Military-grade encryption
- **RS(255,223)**: Corrects up to 16 byte errors per block
- **CGAN LPI**: Waveform statistically identical to AWGN (KS p-value > 0.85)
- **Dual Discriminators**: D2 resets every 25 epochs for dynamic evasion
- **9-Component Loss**: Gaussianity, spectral flatness, cyclostationary suppression
- **Web Interface**: Signal-like chat app with SDR control, spectrum, metrics

---

## 2. Folder Structure

```
LPI_CGAN/
├── models.py                    # Generator, Discriminator, Decoder
├── train.py                     # Basic training (AWGN only)
├── train_radioml.py             # Training with RadioML dataset
├── dataset_radioml.py           # RadioML 2018.01A downloader/loader
├── export_for_grc.py            # Export to TorchScript
├── test_metrics.py              # Verify KS, adversary accuracy, BER
├── requirements.txt             # Python dependencies
│
├── grc/
│   ├── tx_lpi_cgan.grc         # GNU Radio Transmitter flowgraph
│   └── rx_lpi_cgan.grc         # GNU Radio Receiver flowgraph
│
├── webapp/
│   ├── app.py                   # Flask backend
│   └── templates/
│       └── index.html           # Web UI (Signal-like chat)
│
└── README.md                    # This file
```

---

## 3. Phase 0: Install Everything

### 3.1 Create Project Folder

```cmd
mkdir C:\Users\yasht\Desktop\LPI_CGAN
cd C:\Users\yasht\Desktop\LPI_CGAN
```

### 3.2 Install Python Dependencies

**Open Command Prompt (or Anaconda Prompt) and run:**

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy flask flask-socketio pycryptodome reedsolo requests
```

**For GNU Radio specifically (if using radioconda):**
```cmd
conda activate gnuradio
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pycryptodome reedsolo numpy scipy
```

### 3.3 Verify Installation

```cmd
python -c "import torch, gnuradio, numpy, scipy, flask, Crypto, reedsolo; print('ALL OK')"
```

If this prints `ALL OK`, proceed. If any module fails, install it individually.

### 3.4 Install UHD (USRP Drivers)

```cmd
:: Ubuntu/Linux
sudo apt install libuhd-dev uhd-host

:: Windows: Download from Ettus Research website
:: https://www.ettus.com/all-products/ub210-kit/
```

### 3.5 Verify USRP Detection

```cmd
uhd_find_devices
```

You should see your USRP serial number.

---

## 4. Phase 1: Train the CGAN

### 4.1 Option A: Basic Training (AWGN Only, Fast)

```cmd
cd C:\Users\yasht\Desktop\LPI_CGAN
python train.py
```

- Trains on 50,000 synthetic AWGN samples
- Takes 30-60 minutes on CPU
- Good for initial testing

### 4.2 Option B: RadioML Training (Real-World Data, Recommended)

**Step 1: Download RadioML 2018.01A**

```cmd
python dataset_radioml.py
```

This downloads `RML2018.01A.pkl` (~2.5 GB). If automatic download fails:
1. Go to https://www.deepsig.ai/datasets
2. Download `RML2018.01A.pkl` manually
3. Place it in `C:\Users\yasht\Desktop\LPI_CGAN\`

**Step 2: Train with Mixed Dataset**

```cmd
python train_radioml.py
```

**What makes this better:**
- Discriminator sees **real QPSK, QAM16, QAM64, OFDM, PSK, FSK** from actual RF captures
- Your generator must evade detectors trained on **real-world signals**, not just synthetic noise
- 3× data augmentation by windowing each 1024-sample frame into overlapping 512-sample chunks

**Training output:**
```
Epoch 010/200 | D1:-0.342 D2:-0.298 G:4.521 Rec:0.0087
  -> Saved lpi_checkpoint.pt
Epoch 020/200 | D1:-0.298 D2:-0.312 G:3.982 Rec:0.0054
...
```

**When to stop:**
- `Rec` (reconstruction loss) drops below `0.01` — your decoder can recover messages
- `G` stabilizes around 2-5 — generator is converged
- Train for at least 100 epochs for good results

---

## 5. Phase 2: Export for GRC

After training completes:

```cmd
python export_for_grc.py
```

This creates:
- `generator_lpi.pt` — TorchScript model for TX
- `decoder_lpi.pt` — TorchScript model for RX

**Copy these to your GRC project folder.**

---

## 6. Phase 3: Verify Metrics

```cmd
python test_metrics.py
```

**Expected Results:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **KS p-value** | > 0.85 | Cannot reject Gaussian null hypothesis |
| **Adversary Accuracy** | 48-56% | Near-random detection = undetectable |
| **BER** | < 1% | Reliable communication |

**If metrics fail:**
- KS too low → Increase `W_KURT` and `W_VAR` in train script
- Adversary too high → Increase `W_SPEC` and `W_CYCLO`
- BER too high → Train longer or increase `W_REC` to 20.0

---

## 7. Phase 4: GNU Radio TX/RX

### 7.1 Open Transmitter

1. Open **GNU Radio Companion**
2. **File → Open** → `grc/tx_lpi_cgan.grc`
3. **Double-click the Python Block** (`epy_block_0`)
4. Verify parameter `model_path` points to:
   ```
   C:/Users/yasht/Desktop/LPI_CGAN/generator_lpi.pt
   ```
5. Press **F5** (Generate), then **F6** (Run)

**TX Chain:**
```
File Source → Throttle → Unpack K Bits → Chunks to Symbols (-1,+1)
→ Stream to Vector (256) → CGAN Generator → Vector to Stream (512)
→ RRC Filter → Multiply (GLFSR spreader) → USRP Sink
```

### 7.2 Open Receiver

1. **File → Open** → `grc/rx_lpi_cgan.grc`
2. **Double-click the Python Block**
3. Verify parameter `model_path` points to:
   ```
   C:/Users/yasht/Desktop/LPI_CGAN/decoder_lpi.pt
   ```
4. Press **F5**, then **F6**

**RX Chain:**
```
USRP Source → Stream to Vector (512) → CGAN Decoder
→ Vector to Stream (256) → File Sink + QT GUI Time Sink
```

### 7.3 Execution Order

**ALWAYS start RX first, then TX.**

1. Start RX flowgraph (green Play button)
2. Wait 2 seconds
3. Start TX flowgraph
4. Let it run for 10-30 seconds
5. Stop TX, then stop RX

### 7.4 Verify Over-the-Air

Check the recovered file:
```cmd
cd C:\Users\yasht\Desktop\LPI_CGAN
python -c "import numpy as np; d=np.fromfile('recovered_symbols.fc32', dtype=np.complex64); print(f'Recovered {len(d)} symbols')"
```

---

## 8. Phase 5: Web App Interface

### 8.1 Start the Backend

```cmd
cd C:\Users\yasht\Desktop\LPI_CGAN\webapp
python app.py
```

You will see:
```
============================================================
  ShadowComm LPI - Secure Messaging Platform
  Open browser: http://localhost:5000
============================================================
```

### 8.2 Open Browser

Navigate to: **http://localhost:5000**

### 8.3 Web App Features

| Tab | Function |
|-----|----------|
| **💬 Secure Chat** | Type messages, see encryption status, send over SDR |
| **📡 SDR Control** | Change frequency, bandwidth, TX/RX gain, CGAN gain, sample rate |
| **📊 Spectrum** | Real-time FFT waterfall, peak power, spectral flatness |
| **🛡️ LPI Metrics** | KS p-value, adversary detection %, BER, EVM, link quality |
| **⚙️ Settings** | AES passphrase, RS code, CGAN model path, operation mode |

### 8.4 Using the Chat

1. Go to **Settings** tab
2. Enter your **AES Passphrase** (must match on both TX and RX)
3. Go to **SDR Control** tab
4. Set **Center Frequency** (must match on both sides)
5. Go to **Secure Chat** tab
6. Type a message and click **📡 Send**
7. The backend encrypts → RS encodes → CGAN generates → saves waveform
8. In real deployment, the waveform goes directly to USRP

### 8.5 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/config` | GET/POST | Read/write SDR/crypto config |
| `/api/send` | POST | Send encrypted message |
| `/api/receive` | POST | Receive and decrypt message |
| `/api/metrics` | GET | LPI health metrics |

---

## 9. Phase 6: Field Deployment

### 9.1 Hardware Setup

| Component | Recommendation |
|-----------|---------------|
| SDR | USRP B210 (USB) or X310 (Ethernet) |
| Host | Intel i5/i7 laptop or Raspberry Pi 5 |
| Antenna | TX/RX omnidirectional or directional |
| Clock | GPSDO for frequency sync (critical!) |

### 9.2 TX/RX Gain Rules for LPI

| Scenario | TX Gain | RX Gain | Why |
|----------|---------|---------|-----|
| Urban Stealth | 0-5 dB | 30-40 dB | Signal below noise floor |
| Open Field | 10-15 dB | 30-40 dB | Longer range, still covert |
| Emergency | 20-30 dB | 30-40 dB | Reliability over stealth |

### 9.3 Frequency Planning

| Band | Frequency | Use Case |
|------|-----------|----------|
| ISM | 2.4 GHz | License-free testing |
| VHF | 144-148 MHz | Ham radio (with license) |
| UHF | 430-440 MHz | Ham radio (with license) |
| Custom | Your licensed freq | Professional deployment |

### 9.4 Packaging for Field Use

**Option A: Windows Laptop**
```cmd
pip install pyinstaller
pyinstaller --onefile --windowed webapp/app.py
```
Creates `ShadowComm.exe` — double-click to run.

**Option B: Raspberry Pi 5 Field Terminal**
```bash
# On Pi
sudo apt install python3-pip libuhd-dev
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
python3 webapp/app.py --host 0.0.0.0
```
Connect phone to Pi WiFi hotspot, open browser.

**Option C: Docker Container**
```dockerfile
FROM python:3.10-slim
RUN pip install torch numpy scipy flask flask-socketio pycryptodome reedsolo
COPY . /app
WORKDIR /app
CMD ["python", "webapp/app.py"]
```

---

## 10. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: torch` | Wrong Python environment | Use the same Python that runs GRC |
| `FileNotFoundError: RML2018.01A.pkl` | Dataset not downloaded | Run `dataset_radioml.py` or download manually |
| Red errors in GRC Python block | Path issue | Use forward slashes `/` in model_path |
| `U` underruns on USRP | CGAN too slow | Reduce frame_size to 256; use GPU |
| Recovered symbols all +1 | CGAN output too small | Increase gain_factor to 0.5 |
| BER > 5% | Model not trained enough | Train for 200 epochs; increase W_REC |
| KS p-value < 0.5 | Not noise-like enough | Increase W_KURT and W_VAR weights |
| Adversary accuracy > 70% | Detectable features | Increase W_SPEC and W_CYCLO |
| Web app won't start | Port 5000 in use | Change port: `socketio.run(app, port=5001)` |
| USRP not detected | Driver issue | Run `uhd_find_devices`; reinstall UHD |

---

## 11. Complete File Index

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | ~120 | Generator, Discriminator, Decoder architectures |
| `train.py` | ~180 | Basic AWGN training |
| `train_radioml.py` | ~220 | RadioML-enhanced training |
| `dataset_radioml.py` | ~80 | Download & load RML2018.01A |
| `export_for_grc.py` | ~25 | TorchScript export |
| `test_metrics.py` | ~150 | KS test, adversary CNN, BER verification |
| `grc/tx_lpi_cgan.grc` | XML | GNU Radio TX flowgraph |
| `grc/rx_lpi_cgan.grc` | XML | GNU Radio RX flowgraph |
| `webapp/app.py` | ~180 | Flask backend with crypto + CGAN |
| `webapp/templates/index.html` | ~400 | Signal-like web UI |
| `requirements.txt` | 8 | All Python dependencies |

---

## Quick Start (TL;DR)

```cmd
:: 1. Install
cd C:\Users\yasht\Desktop\LPI_CGAN
pip install -r requirements.txt

:: 2. Train
python train_radioml.py

:: 3. Export
python export_for_grc.py

:: 4. Verify
python test_metrics.py

:: 5. Open GRC, load tx_lpi_cgan.grc and rx_lpi_cgan.grc

:: 6. Start web app
cd webapp
python app.py
:: Open http://localhost:5000
```

---

## License & Disclaimer

This system is for **educational and authorized research purposes only**. 
Compliance with local radio regulations is the operator's responsibility.

---

**Built by:** yasht  
**System:** ShadowComm LPI-CGAN v1.0  
**Components:** AES-128-CTR + RS(255,223) + CGAN + USRP
