# RF validation planning checklist (not yet performed)

> **Current evidence is simulation-only.** The Flask app does not drive a USRP, and no over-the-air result has been produced in this workspace. Passing a single offline KS test or one CNN test is not evidence that a transmission is difficult to detect. Do not transmit until the hardware, frequency, power, and test plan are approved for the operating location.

## 1. Offline model handoff

- [ ] Run `python test_metrics.py --ckpt lpi_checkpoint.pt --json-out lpi_metrics.json` and retain the full report.
- [ ] Export with `python export_for_grc.py --checkpoint lpi_checkpoint.pt`.
- [ ] Confirm `cgan_manifest.json` hashes match the deployed TorchScript files.
- [ ] Record the checkpoint, decoder architecture, software versions, sample rate, and exact test seed.
- [ ] Treat the direct generator-to-decoder BER as a noiseless model diagnostic only; it does not predict a hardware link.

## 2. Before any bench RF test

- [ ] Confirm local spectrum rules, licensing, power limits, and test authorization with the responsible operator.
- [ ] Review the SDR and RF-equipment manuals; use a properly rated dummy load, coupler, and attenuation for conducted tests.
- [ ] Verify the attenuation and power budget before connecting transmitter and receiver ports. Do not connect ports directly or use antennas as the first test.
- [ ] Use calibrated instruments and preserve raw captures, configuration, timestamps, and equipment identifiers.
- [ ] Establish a conventional known-signal baseline and a noise-only capture under the same measurement setup.

## 3. Measurements to collect

- [ ] Conducted loopback followed by an authorized radiated test only if the bench test and regulatory review allow it.
- [ ] Record receiver noise floor, capture gain/AGC state, sample rate, center frequency, bandwidth, filtering, synchronization method, and any channel impairments.
- [ ] Measure BER on held-out payloads and report confidence intervals, frame losses, and decoder-correction failures.
- [ ] Compare the generated signal with an independently collected noise baseline using multiple pre-registered features and detectors; document false-alarm and detection probabilities, not accuracy alone.
- [ ] Repeat across independent captures and relevant channel conditions. Keep train/validation/test captures separate.
- [ ] Check spectral emissions, spurious signals, leakage, and occupied bandwidth with calibrated equipment.

## 4. Interpretation

- `ota_verify.py` is an experimental known-message alignment/decoder diagnostic. It is not a certified RF test or a probability-of-intercept assessment; review its assumptions before use.
- A normality/KS result on a fitted or normalized capture is not by itself a valid covertness verdict.
- The in-app spectrum panel is random simulated data, not an SDR measurement.
- Do not claim a covert, secure, or reliable radio link from the current synthetic results. Publish the setup, raw data, detector design, and uncertainty alongside any later RF findings.
