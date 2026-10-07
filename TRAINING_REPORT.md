# CGAN continuation and validation report

**Run date:** 2026-10-07 (UTC timestamps are stored in checkpoint/report metadata)  
**Environment:** CPU-only; PyTorch, deterministic evaluation seed 2026.  
**Scope:** Offline synthetic model diagnostics only. No USRP or OTA capture was used.

## Source material reviewed

- Google Drive `Colab Notebooks` contains `LPI_CGAN_Training.ipynb`, which clones this repository and documents Colab setup and short training experiments.
- Google Drive `LPI_SDR_Project` contains project checkpoints/logs. The available `datasets` folder was empty when checked; no user RF capture dataset was used in this run.
- The selected starting checkpoint is `lpi_checkpoint_v3_610.pt` at generator epoch 610. Generator weights were held fixed during decoder-only tuning. A separate joint generator sweep through epoch 620 was evaluated and rejected (KS p=0.0000, one-CNN accuracy 66.8%, autocorrelation ratio 2.32×, noiseless BER 2.32%).

## Decoder experiments

| Experiment | Validation result | Decision |
|---|---:|---|
| Existing convolutional decoder fine-tune, 2,000 updates | Best BER 2.390% at step 400; baseline 2.766% | Rejected; did not meet the <1% BER screen |
| Global MLP decoder, first 4,000 updates, LR 5e-4 | Best BER 1.814% at step 3,800 | Continued |
| Resume, 10,000 updates, LR 1e-4 | Best BER 1.146% | Continued |
| Resume, 10,000 updates, LR 7e-5 | Best BER 1.079% | Continued |
| Resume, 20,000 updates, LR 5e-5 | Best BER 1.0535% at the script's legacy step label 23,000 | Selected checkpoint weights |
| Focal-loss trial, 20,000 updates, LR 5e-5 | No validation improvement | Rejected; selected weights unchanged |

Adam was restarted for each invocation. The selected weights were reached after 28,000 actual optimizer updates (4,000 + 10,000 + 10,000 + 4,000). Another 36,000 updates after that point (16,000 BCE and 20,000 focal-loss) did not improve the validation BER. The selected checkpoint metadata now records 28,000 optimizer updates; `legacy_step_label` preserves the prior script counter value. The generator remained frozen throughout decoder tuning.

## Final offline metrics

Produced by:

```bash
python test_metrics.py --ckpt lpi_checkpoint.pt --ber-msgs 5000 --json-out lpi_metrics.json
```

| Metric | Result | Project screening threshold | Outcome |
|---|---:|---:|---|
| Pooled marginal KS statistic / p-value | D=0.000682; p=0.2958; 2,048,000 pooled values | p > 0.05 (p > 0.85 aspirational) | Standard screen passes; aspirational p-value not met |
| One freshly trained CNN vs. AWGN | 55.2% accuracy (1,500 train / 1,000 test frames; 4 epochs) | 48–56% | Pass for this one detector only |
| Noiseless generator-to-decoder BER | 13,942 / 1,280,000 bits = 1.0892% | <1% | **Fail** by 0.0892 percentage points |
| Short-lag autocorrelation diagnostic | 2.0739× AWGN reference | Diagnostic only | No formal pass/fail criterion |

The BER is close to, but does not meet, the <1% screen. The KS p-value is sample-size dependent, and a single CNN does not represent all detectors. The autocorrelation calculation is not a formal cyclostationarity test. See `lpi_metrics.json` for full precision and thresholds.

## Export and app verification

- `lpi_checkpoint.pt` contains the epoch-610 generator and the selected `global_mlp_v1` decoder.
- `generator_lpi.pt` and `decoder_lpi.pt` were exported as TorchScript; export smoke checks passed at batch sizes 1 and 3.
- `cgan_manifest.json` records SHA-256 hashes for the checkpoint and exports. `lpi_metrics.json` has the matching checkpoint hash, so the app can display these measured offline values.
- Flask/API smoke tests cover root-page rendering, hash-matched metrics, model health, AES-key redaction, local waveform download, and a multi-frame simulated loopback. The app's default RS(255,191) codewords are byte-interleaved; the test payload round-trips in this no-RF model loopback.

## Limitations and next steps

- No RF capture, SDR calibration, channel, or over-the-air test was performed. The result is not an LPI, security, range, or link-reliability claim.
- The best offline BER remains above target. Improving it further may require model/channel co-design; changing decoder alone has plateaued near 1.05% validation BER.
- Optional GOLD/RadioML data can be used as examples for the threat-model detector if supplied locally, but the Drive dataset directory was empty. Current reported training used synthetic AWGN.
- The browser app is a local demonstration with no authentication. Its AES-CTR wrapper has no per-message nonce or authentication tag and must not be used for sensitive or production data.
