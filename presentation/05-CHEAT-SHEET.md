# CHEAT SHEET — one page, keep it in your pocket

## The six numbers

| | |
|---|---|
| **0.865** | KS p-value vs AWGN (target > 0.05, want > 0.85) |
| **8 / 8** | EW domains passed, composite **Sc = 0.140** (Good; Excellent < 0.10) |
| **55.9 %** | adversary CNN, held out, 400 fit steps (58.8 % at 800 — say both) |
| **3.05e-05** | field BER at 6 dB, with 5 real impairments |
| **0 %** | message FER at every SNR ≥ 2 dB, 64/64 texts exact (bound 4.6 %) |
| **6 / 6** | gate checks PASS, `lpi_eval.py` exit code 0 |

## Frame arithmetic (derive these on the whiteboard)

```
512 payload + 64 pilot = 576 samples
576 / 245760 S/s        = 2.3438 ms per frame
3 frames per AES/RS burst = 7.03 ms, carries 24 characters
128 bits / 2.3438 ms    = 54.6 kb/s gross
10·log10(1024/128)      = 9.03 dB processing gain  (4 chips per bit)
245760 / (2·576)        = ±213.3 Hz unambiguous CFO from the pilot
2 ppm at 2.484 GHz      = ±4.97 kHz  →  coarse acquisition is mandatory (±6 kHz)
```

## The eight EW domains and their pass criteria

`KS p > 0.05` · `|kurtosis − 3| < 0.15` · `entropy > 0.95` · `circularity < 0.15`
· `PAPR dev < 3 dB` · `SCF ratio < 3` · `|C42| < 0.3` · `WVD ratio < 3.7`

Measured: `0.8646 · 2.9907 · 0.9566 · 0.0107 · +0.0087 dB · 1.0129 · −0.0048 · 1.0166`

## Classical waveforms, same benchmark (all fail; we pass 8/8)

| | KS p | detector |
|---|---|---|
| **LPI GAN (ours)** | **0.870** | **55.6 %** |
| BPSK | 0.145 | 100 % (at n = 10 samples!) |
| QPSK | 0.000 | 100 % |
| DSSS | 0.002 | 100 % |
| FHSS | 0.001 | 100 % |
| OFDM | 0.507 | 100 % |
| NRPCS-like | 0.775 | 100 % (detector collapsed — not stealth) |

## `NO LOCK` — exactly four causes, in check order

1. **frequency** — both ends on `2.484e9`?
2. **sample rate** — both ends on `245760`?
3. **gain** — RX gain up, attenuation down, `pilot-SNR > 12 dB`?
4. **key** — session key identical? (wrong key = no correlation peak, by design)

Healthy lock looks like: `frames 6 | CFO +39.5 Hz | pilot-SNR 31.9 dB | dropped 0 | crc 2/2 OK`

## Commands you must know cold

```bash
python -m pytest tests -q -m "not slow"          # 20 passed
python lpi_v4/check_docs.py                      # 82 commands OK
python lpi_v4/lpi_grc.py                         # GRC blocks, no GRC
python lpi_v4/rx_usrp.py --selftest              # TX→channel→RX→text
python lpi_v4/lpi_eval.py --ckpt lpi_v4/run/lpi_v4.best.pt --frames 512 \
       --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400        # the gate
python app.py --snr 6                            # web console, loopback

# radios
python lpi_v4/rx_usrp.py --addr <RX> --freq 2.484e9 --rate 245760 --gain 30 --watch --json-out rx_log.jsonl
python lpi_v4/tx_usrp.py --addr <TX> --freq 2.484e9 --rate 245760 --gain 0 --amp 0.05 --text "ALPHA-INDIA-001" --bursts 20 --loop --period 0.2
```

## The five things to say even if nobody asks

1. Encryption protects content; LPI protects existence. We do both.
2. The stealth is mostly the central limit theorem — an untrained generator
   already passes 8/8. The learning buys invertibility and kills the
   covariance fingerprint.
3. The adversary number belongs to the detector: 55.9 % at 400 fit steps,
   58.8 % at 800. We report both.
4. `0 %` is not a measurement — we quote the Clopper–Pearson bound (FER ≤ 4.6 %).
5. Not yet proved: live UHD streaming at 245760 S/s, PA-on operation, range and
   multipath, and the LO-leakage line that our own receiver uses for CFO.

## The five things to never say

- "It is undetectable." → *"reduced detectability, near-chance, operationally
  challenging."*
- "It is field ready." → *"a validated waveform and a reproducible testbench."*
- "Perfect BER." → *"zero observed errors, 95 % upper bound 1.8e-05."*
- "AES makes it secure." → *"AES protects the payload; the waveform protects
  existence."*
- "Obviously…" / "As everyone knows…" — never.

## If a question destroys a claim

> "You are right — that claim is not supported by what we measured. Thank you."

One sentence. Then move on. Do not defend it.
