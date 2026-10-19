# Demo runbook — print this and keep it on the table

Everything below is a real command from this repository. The commands marked
**VERIFIED TODAY** were run in the workspace on 2026-10-18 and produced the
output shown. The ones marked **BENCH** need the radios and must be rehearsed
tonight on your own hardware.

---

## 1. Table layout

```
        [ projector / big screen ]
                  ▲
   ┌──────────────┴───────────────┐
   │  LAPTOP A (TX)      LAPTOP B (RX)  │   ← deck + web console on A,
   │  tx_usrp.py         rx_usrp.py     │     rx_usrp.py --watch on B
   └────┬───────────────────┬───────┘
        │ USB3              │ USB3
   ┌────┴────┐         ┌────┴────┐
   │ B210 #1 │         │ B210 #2 │
   │ TX/RX   │         │  RX2    │
   └────┬────┘         └────┬────┘
        │  2.4 GHz antenna  │
        └───── 1–3 m ───────┘

  Spare, on the table and powered: a third laptop with the deck and the
  recorded fallback capture. Do not skip this.
```

Cable-loopback rehearsal variant (do this first, before antennas):
B210 `TX/RX` port → **≥ 20 dB attenuator** → `RX2` port. Never connect TX to RX
without an attenuator.

---

## 2. Night before — 40 minutes

Run these in order. Each one is a gate; if any fails, fix it tonight, not
tomorrow.

```bash
cd Final-Project

# 1. dependencies
pip install -r requirements.txt          # or: torch numpy scipy flask flask-socketio pycryptodome reedsolo

# 2. unit tests                                    VERIFIED TODAY: 20 passed, 2 deselected
python -m pytest tests -q -m "not slow"

# 3. docs/CLI drift                                VERIFIED TODAY: OK, 82 commands
python lpi_v4/check_docs.py

# 4. the exact blocks the GRC flowgraphs embed     VERIFIED TODAY: PASS
python lpi_v4/lpi_grc.py
#   expect: frames=72 cfo=-31.5 Hz pilot=24.4 dB text='SHADOWCOMM-v4-GRC-LOOPBA…' -> PASS

# 5. the full metric gate                          VERIFIED TODAY: 6/6 PASS, exit 0, ~4 min
python lpi_v4/lpi_eval.py --ckpt lpi_v4/run/lpi_v4.best.pt --frames 512 \
    --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400 \
    --md lpi_v4/run/eval_arena.md --out-json lpi_v4/run/eval_arena.json

# 6. confirm the model you are about to transmit with is the one you measured
python -c "import hashlib;print(hashlib.sha256(open('lpi_v4/run/lpi_v4.best.pt','rb').read()).hexdigest())"
#   must equal the sha256 in lpi_v4/run/export_manifest.json
```

Then, on the bench (needs radios):

```bash
uhd_find_devices                          # both radios visible?
uhd_fft -f 2.484e9 -s 245760 -g 30        # eyeball the band: DC spike, IQ image, noise floor
```

**Record the outputs.** Screenshot the `6/6 PASS` block and the `uhd_find_devices`
list into a folder called `evidence/`. If a radio dies tomorrow, you show the
panel the evidence you captured last night.

---

## 3. Demo 1 — "Nothing on the spectrum" (90 seconds, the opener)

Do this *before* any explanation. It is the peak the whole five hours is built
around.

**RX box first:**

```bash
python lpi_v4/rx_usrp.py --addr <RX_IP> --freq 2.484e9 --rate 245760 --gain 30 \
    --watch --json-out rx_log.jsonl
```

**TX box second (2 s later):**

```bash
python lpi_v4/tx_usrp.py --addr <TX_IP> --freq 2.484e9 --rate 245760 --gain 0 \
    --amp 0.05 --text "ALPHA-INDIA-001" --bursts 20 --loop --period 0.2
```

**What to say, in this order:**

1. "Two USRP B210s. This one transmits, this one receives."
2. "Look at the spectrum display. There is no bump, no line, no shape."
3. *(wait for the RX line to print)* "…and the message arrived, exactly."
4. "Everything in the next four hours is explaining how both of those things
   can be true at the same time."

**Expected RX output** (this is what a healthy lock looks like — from the
verified offline equivalent):

```
frames    6 | offset 0 | CFO +39.5 Hz (coarse +0.0) | pilot-SNR 31.9 dB | dropped 0
bursts 2 crc 2/2 OK -> 'SELFTEST-SHADOWCOMM-v4-LPI-0042'
text matches exactly -> link OK
```

If you see `frames > 0`, `pilot-SNR > 12 dB` and `crc N/N OK`, the link is
healthy. Those three fields are the entire health check.

---

## 4. Demo 2 — the whole chain on one screen (web console)

This is the "integrated unit" demo. The console was broken and is now wired to
the real v4 code path (see §5).

```bash
python app.py --snr 6                 # loopback, no radio needed
# or, with a real transmitter attached:
python app.py --uhd --addr <TX_IP> --freq 2.484e9 --gain 0
```

Open `http://localhost:5000`.

**Sequence to run on screen:**

1. **Secure Chat** — type `ALPHA-INDIA-001`, press Send.
   The panel shows the ciphertext in hex, then the burst stats
   (`1 burst · 1728 samples · 7.03 ms`), then the decoded message with
   `frames 3 · CFO … Hz · pilot-SNR … dB · CRC 1/1`.
   *Say:* "That hex is what is actually on the air. There is no ASCII in it."
2. **Spectrum** — a flat Welch PSD of the waveform just generated.
   *Say:* "This is the power spectrum of that exact burst. Flat means a
   spectrum analyser has nothing to point at."
3. **LPI Metrics** — the panel now reads the real measurement file.
   *Say:* "Every number on this screen came out of `lpi_eval.py`. The source
   filename and the protocol are printed above them, because a number without
   its protocol is not reproducible."
4. **Settings** — change the session key, resend.
   *Say:* "The key does two jobs: it encrypts the payload and it re-keys the
   Zadoff–Chu preamble. Watch what happens when they disagree."
   → It should print `NO LOCK` / `crc 0/1`. That is a *feature*: wrong key
   cannot even synchronise.

**VERIFIED TODAY, via the Flask test client, no radio:**

```
/api/send    -> status waveform_ready, 1 burst, 1728 samples, 7.03 ms,
                ciphertext 0fd7e8fee42154256641a75d9c182211ee6e8466…
/api/receive -> loopback through the 5-impairment channel @ 6 dB
                frames 3, CFO -203.1 Hz, pilot-SNR 25.8 dB, CRC 1/1, ok true,
                message 'ALPHA-INDIA-001', decode 0.02 s
/api/metrics -> source lpi_v4/run/eval_arena.json, ks 0.8646, adv 55.90,
                BER 3.05e-05, FER 0.0, EW 8/8, Sc 0.1399, gate all pass true
```

---

## 5. What was wrong with the web app, and what changed

Be ready for "did you write this console?" — the honest answer is now a good
one.

| Defect found | Effect | Fix |
|---|---|---|
| `from Crypto.Util.Counter import Counter` | `ImportError` — the app could not start at all | `from Crypto.Util import Counter`; the whole legacy crypto engine replaced |
| `render_template("index.html")` with no `templates/` folder | 500 on `/` | `templates/` added, `template_folder` set explicitly |
| Loaded `generator_lpi.pt` / `decoder_lpi.pt` (v1, 256-bit) | Demo would show the model that has a 2.7 % BER floor | Loads `lpi_v4/run/lpi_v4.best.pt` — the checkpoint the metrics were measured on |
| RS(255,223) at 750 kHz / 32 kS/s | A sample rate a B210 cannot produce; a codec that cannot fit the frame | v4 defaults: 245760 S/s, RS(42,34) |
| AES-CTR with a counter fixed at 0 | Keystream reuse = two-time pad, a total-loss crypto failure | Goes through `lpi_crypto.py`: fresh 8-byte nonce per message |
| `/api/metrics` returned constants (KS 0.91, adv 52.3 %, BER 0.3 %) | A panel displaying numbers nobody measured | Reads `lpi_v4/run/eval_arena.json`, or says "not measured" |
| `spectrum_request` returned `np.random.randn(512)` with a fake peak | A fake spectrum | Real Welch PSD of the waveform generated this session |

Old files are preserved as `app.py.legacy-v1.bak` and `index.html.legacy-v1.bak`.

If asked "why fix it the night before?": because a dashboard that lies is
worse than no dashboard, and the fix was to delete the fake numbers and point
the panel at the measurement file — not to make the numbers look better. The
numbers got *worse* (KS 0.865 not 0.91, adversary 55.9 % not 52.3 %) and true.

---

## 6. Fallback ladder — memorise the order

**Level 1 — the radios do not lock.**
Read the four diagnostics aloud (`pilot-SNR`, `CFO`, `coarse`, `dropped`,
`crc`). Check the four causes in order: frequency, sample rate, gain, key.
Two minutes maximum, then go to Level 2.

**Level 2 — one radio is dead.**
Run the cable-loopback on the surviving B210: `TX/RX` → 20 dB attenuator →
`RX2`. Same commands, one board.

**Level 3 — no radio at all.**
The identical code path, offline, on one laptop:

```bash
python lpi_v4/rx_usrp.py --selftest     # TX -> 5-impairment channel -> RX -> text
python lpi_v4/lpi_grc.py                # the exact blocks the flowgraphs embed
python app.py --snr 6                   # full chain in the browser
```

*Say:* "The radio is not cooperating, so let me show you the same code with
the RF replaced by a calibrated channel model — identical functions, identical
checkpoint. And here is last night's recorded capture from the real hardware."

**Level 4 — even the laptop is dead.**
You still have: the printed cheat sheet, the evidence screenshots from §2, the
`.pptx` on a USB stick, and `lpi_v4/run/eval.md`. A results review does not
require live hardware. Do not apologise twice.

**Rehearse Level 3 tonight.** It is the most likely fallback and it takes 90
seconds to run.

---

## 7. The experiment that turns the demo into data

If you have ten spare minutes and an examiner asks "what is your range?", do
not speculate — measure it:

```bash
# RX box, logging
python lpi_v4/rx_usrp.py --watch --seconds 0.25 --gain 30 --json-out rx_log.jsonl
# TX box, looping
python lpi_v4/tx_usrp.py --loop --period 0.2 --amp 0.05 --bursts 20
```

Walk the antenna from 1 m to 20 m in 2 m steps. Each capture writes one JSON
line with `pilot_snr_db`, `cfo_hz`, `dropped_frames`, `crc_ok`. Plot
`pilot_snr_db` against distance afterwards — **that plot is the measured LPI
range**, and it is a result you can put in the report. `--expect "ALPHA-INDIA"`
turns each capture into a hard pass/fail if you want it in a loop.

---

## 8. Ten minutes before you present

- [ ] Both laptops charged and on mains. Screen sleep **off**.
- [ ] Deck open in full screen on the projector (`F`). Second window: `FINDINGS.md`.
- [ ] `rx_usrp.py --watch` already running on the RX box — starting it live wastes 20 seconds.
- [ ] `tx_usrp.py` command typed into the TX terminal, **not yet executed**.
- [ ] Web console open in a browser tab at `/`, metrics tab loaded.
- [ ] Phones on aeroplane mode. A 2.4 GHz handset joining Wi-Fi will raise your noise floor in front of the panel.
- [ ] Cheat sheet in your pocket. This runbook face-down on the table.
- [ ] One glass of water. You are talking for five hours.
