# LPI-CGAN Field-Test Checklist (One Page)

## >>> v4 (`lpi_v4/`) is the current stack — use the checklist in `lpi_v4/README.md`

Radio: 2.484 GHz, **245760 S/s** (a B210 cannot do 32 kS/s; its floor is ~208 kS/s).
Frame = 512 payload + 64 keyed-Zadoff-Chu pilot samples = 2.34 ms, 128 bits, 9.03 dB
processing gain. One AES/RS burst = 3 frames = 7.0 ms.

```bat
C:\Users\yasht\radioconda\python.exe lpi_v4\v4.bat loop
C:\Users\yasht\radioconda\python.exe lpi_v4\v4.bat eval
C:\Users\yasht\radioconda\python.exe lpi_v4\v4.bat check   :: docs/CLI drift, 1 s
:: cable loopback (>= 20 dB attenuator between TX/RX and RX2):
C:\Users\yasht\radioconda\python.exe lpi_v4\tx_usrp.py --text "ALPHA-INDIA-001" --bursts 20 --loop --period 0.2 --rate 245760 --gain 0
C:\Users\yasht\radioconda\python.exe lpi_v4\rx_usrp.py --watch --seconds 0.25 --gain 30 --rate 245760 --json-out rx_log.jsonl
:: antennas: same, raise tx_gain 0 -> 20, expect pilot-SNR > 15 dB
```

Before the first transmit, check that the TorchScript in `lpi_v4\run\` is the model
you measured: `run/export_manifest.json` stores the `sha256` of the checkpoint it was
exported from, so `python -c "import hashlib;print(hashlib.sha256(open(r'lpi_v4/run/lpi_v4.best.pt','rb').read()).hexdigest())"`
has to print that same digest. (It is also `pytest`'s `test_exported_torchscript_matches_the_checkpoint`.)
Related trap: the *legacy* `rx_lpi_cgan.grc` loads `decoder_lpi.pt` from
`C:\Users\yasht\Desktop\LPI_CGAN\` -- v1's 256-bit model. Same filename, different
architecture; the v4 flowgraphs load `run/lpi_v4.best.pt` through `lpi_grc.py` instead.

Expected on every capture: `frames > 0`, `pilot-SNR > 12 dB`, `bursts N crc N/N OK`,
`CFO` within +-6 kHz (found automatically), text matching exactly. Pass criteria are
unchanged (BER < 1 % cable, < 5 % antenna, KS p > 0.05, adversary 48-56 %) and v4
meets them in simulation -- see the table in `lpi_v4/README.md`. What to *record* from
`rx_log.jsonl`: `pilot_snr_db`, `cfo_hz`, `dropped_frames`, `crc_ok` vs distance/gain.

---

## (legacy v1-v3 checklist, kept for the older flowgraphs)

Project folder: `C:\Users\yasht\Desktop\LPI_CGAN` | Python: `C:\Users\yasht\radioconda\python.exe`
Radio: one USRP, TX = TX/RX port, RX = RX2 port (B210 full-duplex), 750 MHz, 32 kS/s.

---

## Step 0: Confirm the model is ready (after the sweep finishes)

- [ ] `run_sweep.bat` completed and produced `lpi_checkpoint.pt` (full-PASS snapshot) + `generator_lpi.pt` / `decoder_lpi.pt` (auto-exported)
- [ ] Quick self-check: `python test_metrics.py --ckpt lpi_checkpoint.pt` -> all four PASS (KS > 0.05, adversary CNN ~50%, BER < 1%)
- [ ] If the sweep is not done yet: the flow still works with the current `lpi_checkpoint_v3_610.pt`, but BER will read ~4% (decoder floor)

## Step 1: Cable loopback (cable + attenuator first — not antennas)

- [ ] USRP TX/RX port -> attenuator (>= 20 dB) -> coax cable -> RX2 port
- [ ] Open `grc\rx_lpi_cgan.grc` in GNU Radio and run it first (noise waveform on the right; `rx_raw.fc32` starts recording)
- [ ] Open `grc\tx_lpi_cgan.grc` in GNU Radio and run it (transmission starts; `tx_bits.uint8` records the transmitted bits)
- [ ] Run for **60 seconds**, stop TX first, then stop RX
- [ ] Offline verification (from `C:\Users\yasht\Desktop\LPI_CGAN`):

```bat
C:\Users\yasht\radioconda\python.exe ota_verify.py --raw rx_raw.fc32 --msg test_message.txt --decoder decoder_lpi.pt
```

- [ ] Expected: offset/CFO/phase locked automatically, **BER < 1% PASS**, KS p > 0.05 PASS

## Step 2: Over-the-air with antennas (after the cable passes)

- [ ] Two antennas (or one TX / one RX), 1-3 m apart, raise `tx_gain` from 10 to 30-40
- [ ] Repeat the TX/RX + verification flow from Step 1
- [ ] Expected: BER may rise to 1-5% (multipath + SNR variation is normal); KS should still PASS

## Pass Criteria

| Metric | Pass threshold | Meaning |
|---|---|---|
| BER | < 1% (cable) / < 5% (antenna) | Covert link is usable |
| KS p | > 0.05 | **The radiated signal is statistically indistinguishable from noise (the core LPI result)** |
| offset / CFO / phase | Locked automatically by the script | Lock failure = capture too short or SNR too low |

## Troubleshooting

| Symptom | Most likely cause |
|---|---|
| CFO not found (low line SNR) | TX not running / too much attenuation / frequency mismatch (750 MHz); the script automatically falls back to a slow grid search — let it finish |
| Locked but BER ~50% | Wrong start order (RX must start before TX); `rx_raw.fc32` too short (< 32 frames) |
| Locked but BER elevated | SNR below ~15 dB / decoder is still the v3_610 floor version -> wait for the sweep and use the final model |
| `decoder_lpi.pt` fails to load | Model not exported -> run `python export_for_grc.py` manually (needs `lpi_checkpoint.pt`) |
| `[CGAN Dec] Loaded on cpu` then nothing in GRC | Normal — the inline decoder is only a monitor; official results come from `ota_verify.py` |

## LPI Notes (points to include in your report)

- **LO leakage betrays the transmission**: the USRP's TX LO leaks a spectral line exactly at the CFO (the verification script uses it to estimate the CFO). An eavesdropper can see the same line -> a real LPD-level vulnerability. Mitigations: keep TX gain at the minimum, use a well-calibrated board, and discuss it honestly in the report.
- **The KS test is the eavesdropper's perspective**: an adversary does not need the bits — only a statistical test at the level of KS / adversary CNN. PASSing those two is the evidence of covertness; BER is only usability.
- The inline decode (`recovered_symbols.fc32`) is for demonstration; **official data always goes through `rx_raw.fc32` + `ota_verify.py`**, which performs normalization, the three-parameter search, and per-frame phase tracking.

## File Quick Reference

| File | Purpose |
|---|---|
| `run_sweep.bat` | Training sweep until all four metrics PASS (run overnight) |
| `grc\tx_lpi_cgan.grc` / `grc\rx_lpi_cgan.grc` | TX / RX flowgraphs (rebuilt; originals in `.bak`) |
| `rx_raw.fc32` | Raw baseband recorded by the RX (verification input) |
| `ota_verify.py` | Offline verification: parameter lock + BER + KS report |
| `test_message.txt` | Known 120 KB test message (ground truth) |
| `lpi_checkpoint_v3_610.pt(.sweep.csv)` | Sweep starting point and log |
