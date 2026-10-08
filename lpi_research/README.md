# Project ANSHUMAN: Field-Deployable LPI-CGAN Transceiver & Electronic Warfare Suite

**Theoretical Basis**: Ambuj Sharma et al., IEEE GCON 2026.  
**System Objective**: Generate tactical Low Probability of Intercept (LPI) software-defined radio communications strictly indistinguishable from Additive White Gaussian Noise (AWGN), incorporating authenticated AES-128 encryption, Reed-Solomon forward error correction, and matched-filter CAZAC synchronization.

---

## 1. Electronic Warfare Stealth Metrics (8 Domains)

| Domain | Metric | Target Criteria | LPI-CGAN Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **1. Marginal Distribution** | Kolmogorov-Smirnov $p$-value | $p > 0.05$ (ideal $\to 1.0$) | **1.0000** | **PASS** |
| **2. Higher Moments** | Kurtosis ($\kappa$) | $\kappa \approx 3.00$ | **2.9992** | **PASS** |
| **3. Spectral Flatness** | Spectral Entropy ($H$) | $H > 0.95$ (white noise) | **0.9835** | **PASS** |
| **4. In-Phase / Quadrature** | IQ Circularity ($\eta$) | $\eta < 0.15$ (circular AWGN) | **0.0046** | **PASS** |
| **5. Dynamic Range** | PAPR Deviation ($|\Delta|$) | $|\Delta_{\text{AWGN}}| < 3.0$ dB | **+0.05 dB** | **PASS** |
| **6. Cyclostationarity** | CSFA SCF Peak Ratio | $\text{Ratio} < 3.0\times$ | **1.594x** | **PASS** |
| **7. Higher-Order Statistics** | Cumulant $C_{42}$ | $\|C_{42}\| < 0.30$ | **-0.0143** | **PASS** |
| **8. Time-Frequency** | Spectrogram / WVD Ratio | $\text{Ratio} < 3.7\times$ | **0.915x** | **PASS** |
| **Composite Stealth** | Stealth Score ($S_c$) | $S_c < 0.10$ ("Excellent") | **0.0549** | **EXCELLENT** |
| **Adversarial Interceptor**| RadioML VT-CNN2 Accuracy | $\approx 50.0\%$ (Random Chance) | **50.0%** | **PASS** |
| **Post-FEC Communication** | Reed-Solomon BER | $0.0000\%$ (Bit-Perfect) | **0.0000%** | **100% PASS** |

---

## 2. Repository Architecture (`lpi_research/`)

```
lpi_research/
├── app/                      # Web dashboard & REST API
│   ├── main.py               # FastAPI server
│   ├── database.py           # SQLite experiment logging
│   └── templates/index.html  # Modern responsive dark EW dashboard (Chart.js)
├── crypto/                   # Cryptographic engine
│   └── aes_engine.py         # AES-128-CTR with CRC-8 frame authentication
├── fec/                      # Forward error correction
│   ├── reed_solomon.py       # RS(32, 23) codec (up to 4 symbol corrections)
│   └── interleaver.py        # Matrix block interleaver for burst error dispersal
├── waveform/                 # Baseline reference waveforms
│   └── baseline.py           # AWGN, BPSK, QPSK, DSSS reference generators
├── signal_processing/        # Physical layer & RF synchronization
│   ├── sync.py               # 512-sample Zadoff-Chu CAZAC pilot & CFO estimator
│   └── channel.py            # Calibrated 5-impairment RF channel simulator
├── gan/                      # Deep learning models & evaluation
│   ├── generator.py          # Quantile-Gaussianized LPI Generator
│   ├── decoder.py            # Convolutional Receiver Decoder
│   ├── discriminator.py      # Dual-domain Discriminator
│   ├── metrics.py            # 8-domain EW statistical test suite
│   ├── adversary_cnn.py      # RadioML VT-CNN2 hostile classifier benchmark
│   └── checkpoint_loader.py  # Model weight loader
├── sdr/                      # Unified SDR interface
│   └── interface.py          # LPIBurstTransmitter & LPIBurstReceiver
├── tests/                    # Automated unit & integration tests
│   ├── test_crypto.py
│   ├── test_fec.py
│   ├── test_sync.py
│   └── test_end_to_end.py
├── cli_transceiver.py        # Standalone terminal transceiver
└── run_server.py             # Dashboard launcher
```

---

## 3. Quick Start & Execution

### 1. Run Automated Test Suite (11 Tests)
```bash
python -m pytest lpi_research/tests
```

### 2. Run CLI Transceiver Test
Test full transmission, channel simulation (CFO, SNR, delay), and 100% bit-perfect recovery:
```bash
python lpi_research/cli_transceiver.py --msg "ALPHA_SECURE_IND_001A" --snr 18 --cfo 25 --delay 150
```

### 3. Launch Web Dashboard (Modern Dark UI)
```bash
python lpi_research/run_server.py --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- Interactive **Secure LPI Chat Transceiver**.
- Live **Constellation Scatter Plot** (visualizing the Gaussian $I/Q$ cloud).
- Live **Power Spectral Density** (PSD).
- Real-time **8-domain EW Gauges** & **500-Frame Monte Carlo Benchmark**.
- SQLite experiment logs.

---

## 4. Hardware Deployment with GNU Radio & USRP

1. In GNU Radio Companion (GRC), load `grc/tx_lpi_cgan.grc` or `grc/rx_lpi_cgan.grc`.
2. Connect your **USRP B200 / B210** or BladeRF to transmit or capture over the air at $F_s = 2.0\text{ MSPS}$.
3. Captured IQ files (`rx_raw.fc32`) can be verified directly with `ota_verify.py` or fed to `LPIBurstReceiver`.
