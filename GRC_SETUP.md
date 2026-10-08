# GNU Radio / USRP commissioning guide

> **Status:** The `.grc` files are model-integration lab scaffolds, not validated transceivers. The TX Epy block is disabled by default. The RX graph has no preamble detection, frame timing, carrier-frequency/phase recovery, or channel equalization. No hardware or OTA result has been validated in this repository.
>
> Do not radiate until you have confirmed authorization, a permitted frequency, the exact radio's tuning limits, and an appropriate test setup. The placeholder center frequency is `0` so it must be explicitly configured; the previous `750000` Hz value was not a recommendation. The TX Epy gate outputs zero baseband while disabled, but it is **not** a hardware RF interlock and does not guarantee zero LO leakage.

## 1. What is integrated

The TX graph in `tx_lpi_cgan.grc` takes a byte file, unpacks bits, maps 0/1 to BPSK symbols `-1/+1`, groups 256 symbols, calls the TorchScript generator, then streams 512 complex IQ samples to the UHD sink. The generator signature is `(z: Bx64, bits: Bx256) -> IQ: Bx2x512`.

The RX graph in `rx_lpi_cgan.grc` records raw complex samples, groups the stream into fixed vectors of 512, calls the TorchScript decoder, and writes 256 hard bit symbols. The decoder signature is `(IQ: Bx2x512) -> probabilities: Bx256`.

The GRC graph currently transports raw test-file bits. It does **not** implement the Flask demo's AES/RS encoding, a packet header, a preamble, or robust frame synchronization. The decoded output is complex `-1/+1` symbols, not yet a recovered text file.

## 2. Software setup

1. Check out the pushed session branch and enter the repository root:

   ```bash
   git fetch origin
   git switch arena/d1ad1323-final-project
   git pull --ff-only
   ```

2. Use the Python environment that launches GRC/UHD. In Radioconda, for example, activate the environment you use for GNU Radio and check that it can import both GNU Radio and PyTorch:

   ```bash
   python -c "import sys, torch; from gnuradio import gr; print(sys.executable); print(torch.__version__)"
   uhd_find_devices
   ```

   Installing PyTorch into a different Python than GRC uses will not make it available to the Epy blocks.

3. From this repository root, export and smoke-test the deployed models:

   ```bash
   python export_for_grc.py --checkpoint lpi_checkpoint.pt
   ```

   This writes `generator_lpi.pt`, `decoder_lpi.pt`, and `cgan_manifest.json`. The export script checks model shapes at batch sizes 1 and 3.

## 3. Configure TX in GRC

1. Open `tx_lpi_cgan.grc` in GNU Radio Companion.
2. Leave the Epy block's `tx_enable` set to `False` while configuring and inspecting the graph.
3. Set the Epy block's `model_path` to the full local path of `generator_lpi.pt` if GRC is not launched from the repository root. Use a path valid on your OS.
4. Set the File Source to a local, non-sensitive test file. The graph's file paths are relative to its working directory; either launch GRC from the repository root or set absolute paths for the source and sinks. The current source repeats the file indefinitely; the test message is not encrypted by this GRC path.
5. Set the UHD sink device address/serial and antenna according to your exact USRP manual. Do not assume the default antenna port is correct for your model.
6. Set `center_freq` only to a frequency that is both authorized for your test and supported by your radio. Set `samp_rate` to a rate supported by the device and driver. The values in these fields are not validated by the model.
7. Keep `tx_gain` at the lowest setting required for the authorized conducted test. There is no universal safe gain value. The `gain_factor` in the Epy block only scales digital IQ; it is not an RF power setting.
8. GRC's `tx_enable=False` makes the Epy block emit zeros. After verifying the authorized setup and using the approved conducted arrangement, you may set `tx_enable=True` for that test. This software flag does not disable the UHD device or guarantee no RF leakage.

## 4. Configure RX in GRC

1. Open `rx_lpi_cgan.grc`.
2. Set the Epy block `model_path` to the local `decoder_lpi.pt` path.
3. Configure the UHD source device address, antenna, center frequency, sample rate, and gain for your radio and test. TX and RX must use matching channel settings, but receiver gain must be calibrated rather than guessed.
4. Change `rx_raw.fc32` and `recovered_symbols.fc32` if you want output elsewhere.
5. `remove_dc` and `normalize_rms` default to `False`, matching the model-to-model evaluator's raw-frame convention. Changing them may change BER; test and document each setting.
6. Do not expect successful radio decoding yet: the graph slices every 512 incoming samples without detecting frame starts and does not correct CFO/phase/timing offsets. Add a known preamble/access sequence and a synchronizer before the decoder for any channel test.

## 5. Safe validation sequence

1. **Software only:** run the export smoke test and `python -m unittest discover -s tests -v`.
2. **No RF:** save TX baseband to a file instead of the UHD sink, if available in your GRC version, and confirm frame shape/rate with GNU Radio tools.
3. **Conducted bench test only:** follow the hardware manuals and lab RF procedure. Use a properly rated attenuator/coupler and suitable load between TX and RX. Never cable TX directly to RX. Verify the attenuation and power budget before enabling the transmitter; do not start with antennas.
4. Add framing/synchronization and capture raw RX IQ. Compare recovered bits only after aligning the known transmitted bitstream and removing headers/padding. Record device model/serial, software versions, sample rate, center frequency, gains, attenuator, and capture conditions.
5. Consider an OTA test only when authorized by the responsible lab/operator and local spectrum rules. Use calibrated measurements and a matched noise-only baseline. A single KS p-value or CNN accuracy is not an LPI verdict.

## 6. Current model limits

The attached offline report (`lpi_metrics.json`) is generated from synthetic model samples, not a USRP capture. Its noiseless BER is 1.0892%, above the repository's <1% screening threshold. The marginal KS and one-CNN checks are limited diagnostics. `ota_verify.py` is experimental and should not be treated as a calibrated receiver or an OTA/LPI certification tool.

## 7. What is needed for a hardware-specific setup

To tailor the device-address, antenna, rate, and test procedure, record the exact USRP model/revision, host OS, GNU Radio/UHD version, whether you have one or two radios, and the authorized test arrangement/frequency supplied by your lab. Do not infer a frequency or transmit power from the example graphs.
