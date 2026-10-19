# PROJECT ANSHUMAN — slide-by-slide

Generated from `presentation/slides_data.py` (the single source of truth).
43 slides. Speaker notes and the 10-year-old analogies are embedded in
both `ANSHUMAN_deck.html` and `ANSHUMAN_deck.pptx`.

### 1. PROJECT ANSHUMAN

*? min*


**Speaker notes:** Good morning. Before I explain anything, I am going to show you something. On this table are two USRP B210 radios. In about four minutes I will send an encrypted message between them, and I will show you that on a spectrum analyser that transmission is indistinguishable from the noise floor of an empty room.

**For a 10-year-old:** We built a radio that whispers so quietly that nobody can tell it is talking.

### 2. First: the 90-second demo

*? min*

- TX box: plaintext → AES-128-CTR → CRC-8 → RS(42,34) → CGAN → IQ → B210
- RX box: B210 → keyed sync → CFO/phase → decoder → RS → AES → plaintext
- >> Look at the waterfall: there is no bump, no line, no shape to find
- >> Then the exact text appears on the receiver screen

**Speaker notes:** Do not narrate the chain yet. Just say: this is the transmitter, this is the receiver, watch the spectrum. Send 'ALPHA-INDIA-001'. Let the room see the flat waterfall and then the decoded text. That contrast -- nothing on the spectrum, message received -- is the whole project in one image. Everything in the next four hours is explaining how that is possible and how far it is from being trustworthy.

**For a 10-year-old:** Look: the picture shows nothing. But the message still arrived.

### 3. The one-slide result

*? min*


**Speaker notes:** Six numbers. KS p-value 0.865 means a statistician looking at my samples cannot reject the hypothesis that they are pure noise. Eight out of eight electronic-warfare test domains pass. A convolutional detector trained specifically on my waveform gets 55.9 percent -- a coin flip is 50. Bit error rate three in a hundred thousand at 6 dB with carrier offset, phase noise, DC offset and IQ imbalance all switched on. And every message came back exactly right. The last card is the one I am most proud of: it is a gate that exits non-zero if any of this regresses, so the numbers cannot silently rot.

**For a 10-year-old:** Five checks say 'this looks like noise'. One check says 'the message still got through'. Both are true at the same time.

### 4. What this project is, in one breath

*? min*

- A military radio must do two things at once: deliver the message, and not announce that it exists.
- Encryption solves the first problem. It does nothing for the second -- the enemy does not need your words, only your position.
- >> Encryption protects CONTENT. LPI protects EXISTENCE.
- We make the transmitted waveform statistically identical to background thermal noise, across eight independent detection domains, while still carrying an AES-encrypted, error-corrected payload.
- -- Method: a conditional GAN learns a keyed spreading codebook; the message rides inside 128 chips per frame; the receiver is a matched filter plus a learned head.
- -- Proof: statistical tests, a trained enemy CNN, per-SNR BER/FER, and over-the-air hardware.

**Speaker notes:** Slow down here. Officers in the room already know the kill chain; academics want the mechanism. This slide serves both. The single most quotable line is 'encryption protects content, LPI protects existence' -- say it twice, once now and once in the conclusion.

**For a 10-year-old:** A lock keeps people from reading your letter. But they can still see that a letter was sent, and where it came from. We hide the letter itself.

### 5. How a radio gets caught

*? min*


**Speaker notes:** This act is the 'why'. Keep it to eight minutes. It is the part that makes the rest matter.

### 6. The intercept kill chain

*? min*

- 1. Energy detection -- is there power above the noise floor?
- 2. Direction finding -- triangulate the emitter, get a grid square
- 3. Traffic analysis -- even with no words: who talks, when, how often, for how long
- 4. AI classification -- a CNN on I/Q or a spectrogram says 'this is BPSK, this is OFDM'
- 5. Jam or intercept -- targeted denial, or a recorded capture for offline attack
- >> Steps 1 to 3 need nothing from your cryptography. Step 4 needs ten samples.
- Reference: BPSK reaches 100 % detection accuracy at n = 10 training samples.

**Speaker notes:** The point to land: every one of these five steps happens *before* anyone touches your ciphertext. AES is irrelevant to all of them. Quote the BPSK n=10 number -- it is from the benchmark table and it is the number that makes officers lean forward.

**For a 10-year-old:** A spy does not need to read your note. They just need to see the light on in your window.

### 7. Encryption is not covertness

*? min*


**Speaker notes:** If an officer asks 'but we already use AES on our radios', this is the slide. The two are complementary, not alternatives. Say explicitly: we use AES too -- AES-128-CTR with a fresh nonce per message, plus CRC-8 and Reed-Solomon. We are adding the layer that was missing, not replacing the one that works.

**For a 10-year-old:** Locking your diary stops people reading it. It does not stop them seeing you write in it every night at 9 pm.

### 8. LPI, LPD, LPE — and the law we have to obey

*? min*

- LPI — Low Probability of Intercept: the enemy cannot detect/classify the transmission
- LPD — Low Probability of Detection: the enemy cannot tell signal from noise at all
- LPE — Low Probability of Exploitation: even if detected, it cannot be decoded or geolocated usefully
- -- We target LPI/LPD by statistics, LPE by keying (the spreading codebook is derived from the session key)
- >> Bash et al. square-root law: covert throughput grows like √n, not n. Covert is intrinsically low-rate.
- Our honest rate today: 128 bits per 2.34 ms frame = 54.6 kb/s gross, 24 characters per 7.0 ms burst net.

**Speaker notes:** Academics will ask about the square-root law. The honest answer is: yes, covert capacity is fundamentally sub-linear, and that is exactly why our payload is 24 characters per burst and not streaming video. We are not claiming a broadband covert link; we are claiming a short-burst covert text link, which is what a patrol actually needs.

**For a 10-year-old:** There is a law of physics that says: the more you hide, the less you can say. We obey it.

### 9. How we got here

*? min*


**Speaker notes:** This is the act the examiners care about most, because it is the act that shows work. Tell it chronologically and do not skip the failures. Roughly 45 minutes.

### 10. Step 0 — Literature and the gap

*? min*

- Pillar 1 — End-to-end deep learning for the physical layer (O'Shea & Hoydis 2017): a network can jointly learn transmitter and receiver.
- Pillar 2 — Conditional GAN (Mirza & Osindero 2014): condition a generator on a label. We condition it on the message bits themselves.
- Pillar 3 — Covert capacity, the √n law (Bash et al. 2013): the ceiling any LPI design must respect.
- >> The gap: no published system combined (a) a CGAN waveform, (b) multi-domain EW evaluation, (c) few-shot detector resistance, and (d) commercial-SDR over-the-air validation with encryption.
- That four-way intersection is the contribution.

**Speaker notes:** Be precise about the novelty claim. It is not 'we invented GANs' and not 'we invented LPI'. It is the *simultaneous* demonstration of four things that prior work each demonstrated alone. Table 9 of the Elsevier draft is the comparison that supports this -- seven prior approaches, none of which has all four.

**For a 10-year-old:** Other people built the engine, the wheels and the steering wheel separately. We were the first to build a car and drive it.

### 11. Step 1 — GNU Radio: the first transmission

*? min*

- First flowgraph (`GNU 1.grc`): Signal Source → Throttle → QT Time Sink. A tone, in software, no radio.
- Then `USRP testing.grc`: Signal Source → UHD USRP Sink, and UHD USRP Source → QT Frequency Sink.
- >> The moment the project became real: a peak we created in software appeared on the FFT of a second radio across the room.
- What that proved: sample rates, gains, frequencies and cabling are all real constraints, and the FFT is the truth-teller.
- First mistake everyone makes: TX gain too high, RX saturated, or TX/RX frequency off by the LO offset.

**Speaker notes:** Tell this as a story, not a list. 'The first time a peak that existed only as a number in a Python block appeared on a spectrum display on another machine, that is when the project stopped being homework.' Then be concrete: the knobs you had to learn were centre frequency, sample rate, TX gain, RX gain and antenna port. Keep `GNU 1.grc` and `USRP testing.grc` open in GRC to show them.

**For a 10-year-old:** First we made the radio say 'beep'. Then another radio heard the beep. That was step one.

### 12. Step 2 — The test bench: a real DSP chain

*? min*

- File Source (bits) → Unpack k Bits → Chunks to Symbols → BPSK map (−1, +1)
- → Root-Raised-Cosine filter (pulse shaping, band-limiting)
- → Multiply by a GLFSR sequence (direct-sequence spreading)
- → UHD USRP Sink  |  UHD USRP Source → the mirror chain → File Sink
- Python blocks (`TESTbench_epy_block_0..3.py`) carried the parts GRC has no block for
- >> This bench is why we could later say 'the AI is not the reason it works' -- we had a working conventional link first.

**Speaker notes:** This is the slide that establishes credibility with the RF people. You built a conventional spread-spectrum BPSK link *before* you touched a neural network, so you know what a good link looks like and you have something to compare the waveform against. Emphasise the RRC filter and the GLFSR spreader -- those are textbook, and you can point at them in the flowgraph.

**For a 10-year-old:** Before teaching a robot to whisper, we learned to speak properly ourselves.

### 13. Step 3 — The payload: AES + CRC + Reed–Solomon

*? min*

- plaintext (≤ 24 B) → fresh 8-byte nonce + AES-128-CTR ciphertext → CRC-8 = 34 data symbols
- → RS(42,34): 8 parity symbols, corrects up to 4 bad bytes = 42 symbols = 336 bits
- → 3 covert frames of 128 bits (384 bits, 48 bits of zero padding at the end)
- >> Three bugs this layer had to fix, all of them silent:
- -- (a) the reference draft pads 336 bits into 256 -- that destroys 80 payload bits, so a 'BER = 0' claim cannot be true for the whole message. We derive the frame count instead.
- -- (b) the web app encrypted a string while the GRC chain modulated raw file bytes -- TX and RX were not speaking the same protocol.
- -- (c) AES-CTR with a counter fixed at 0 is a two-time pad: keystream reuse is total loss. The nonce is now per message.
- GCM mode also implemented (20 B payload, 4-byte truncated tag) for authenticated encryption.

**Speaker notes:** This is a strong slide for academics because it shows you read the specification critically rather than implementing it blindly. The 336-to-256 truncation bug is the kind of finding that separates a project from a copy. Say it plainly: we found a bug in the design we were given, and here is the line of code that fixes it. Note the RS choice: RS(42,34) with nsize=42 -- RS(255,223) does not fit a 42-byte codeword, and reedsolo silently returns ('', ok=True) if you get that wrong.

**For a 10-year-old:** We lock the message, add a checksum, then add spare letters so a few damaged ones can be rebuilt.

### 14. Step 4 — The C-GAN: what it actually is

*? min*

- GENERATOR — takes latent noise z (64-dim) + the 128 message bits, outputs 2×512 real samples (512 complex).
- -- Its core is a *learned keyed spreading codebook*: a 128 → 1024 linear layer, one code vector per bit.
- -- A small conv head (gain 0.2) refines the result to remove residual structure.
- DISCRIMINATOR — sees a frame and answers 'noise or message?'. Two of them; D2 is reinitialised every 25 epochs so the generator cannot memorise one boundary.
- DECODER — the friendly receiver. A matched filter against the same codebook, then a learned conv head that outputs one logit per bit.
- >> The weights *are* the key. Same architecture, wrong key → the preamble will not even lock.
- Training: 500 epochs, batch 128, Adam (β = 0.5, 0.999), η_G = 2e-4, η_D = 1e-4, on a Colab T4.

**Speaker notes:** Do not let this become a machine-learning lecture. The one idea that matters: the generator is not free-form. It is a *spreading codebook*, so every transmitted sample is the sum of 128 independent keyed chips -- and a sum of many independent things is Gaussian. That is the central limit theorem doing the stealth, and it is why the stealth is robust instead of tuned. If a professor asks 'is this just DSSS with extra steps', the answer is: it is DSSS where the codes are learned jointly with the receiver and shaped against a trained detector -- and that difference is what removes the cyclostationary signature.

**For a 10-year-old:** Imagine 128 people each whispering one letter of your message at the same time, all in different voices. To an outsider it is just a crowd noise. Your friend knows each voice, so she can pull each letter back out.

### 15. Step 4b — The nine-component loss

*? min*


**Speaker notes:** Power normalisation has the highest weight because energy detection is the cheapest and most sensitive EW tool an enemy has -- a radiometer is a diode and an integrator. Also mention the v4 addition that is *not* in the nine: `loss_cov`, a covariance/Gram-spectrum term. Any generator whose internal dimension is smaller than the sample count produces frames in a fixed low-rank subspace, and the eigenvalue detector family used in cooperative spectrum sensing sees exactly that. It is not visible in any of the eight per-frame domains.

**For a 10-year-old:** Nine different referees, each checking one thing: too loud, too peaked, too rhythmic, too samey. We had to satisfy all nine at once.

### 16. Why not DSSS / FHSS / QPSK / OFDM?

*? min*


**Speaker notes:** This is the slide that answers the question you will definitely get. Read it in words, not numbers: every classical waveform is separated from noise with 100 percent accuracy by a trained detector. Ours sits at 55.6 percent, and 50 is a coin flip. Then explain *why* they fail, because that is the insight: BPSK has a huge IQ circularity of 0.735 because it only ever sits on two points of the complex plane; DSSS and FHSS still have periodic structure, so a cyclostationary detector finds the code rate or the hop rate; OFDM has a cyclic prefix that is a gift to a detector. Note the NRPCS footnote honestly: it scores 50 percent only because its detector collapsed to predicting 'signal' for everything -- that is not stealth.

**For a 10-year-old:** Every other trick still leaves a footprint in the sand. Ours leaves a footprint that looks exactly like the sand.

### 17. Step 5 — How we hit the AWGN statistics

*? min*

- Target: amplitude marginal Gaussian, flat PSD, zero autocorrelation, kurtosis 3, IQ circularly symmetric.
- We expected to *tune* our way there with nine loss weights. We measured something better.
- >> An entirely untrained generator (random weights, 512 frames) already scores KS p 0.857, kurtosis 2.998, entropy 0.959, 8/8 domains, and 1.6e-4 BER through the field path.
- So the Gaussianity comes from the central limit theorem over 128 keyed chips per frame, plus the ε-blend — it is free.
- What the losses actually buy: keeping it Gaussian *while* the waveform is made invertible, and killing the covariance-domain fingerprint.
- >> The lesson we now teach: do not tune for KS. Tune the two things that fight each other — ε (masking) and the reconstruction weight — and let the CLT do the rest.
- ε is the single stealth↔reliability knob: x = √(1−ε)·s + √ε·n₀. Shipped value 0.10.

**Speaker notes:** This is the most intellectually honest slide in the deck and academics will respect it enormously. You are telling them that your headline metric is largely explained by a theorem, not by your tuning -- and that you found this by running the control experiment. It also protects you: if someone says 'your stealth is trivial', you have already said it, and you have already explained what the learning actually contributes (invertibility plus the Gram/covariance domain).

**For a 10-year-old:** If you add up 128 random little pushes, you always get a bell curve. That is a 300-year-old theorem, and it is doing our hiding for free.

### 18. Step 6 — The eight-domain EW evaluation

*? min*

- 1 KS p-value on the I/Q amplitude marginal — distributional
- 2 excess kurtosis (target 3.0 ± 0.15) — distributional
- 3 spectral entropy > 0.95 (Welch/Hann) — spectral
- 4 IQ circularity < 0.15 — second-order
- 5 PAPR deviation < 3 dB — envelope / energy
- 6 SCF ratio < 3 (cyclostationary feature analysis) — cyclostationary
- 7 HOS cumulant C42 < 0.3 — higher-order
- 8 WVD time–frequency ratio < 3.7 — time-frequency
- >> Composite Sc = weighted mean deviation, equal weights. 0.140 = 'Good' (Excellent is < 0.10). It never touches training — it is a reporting heuristic only.
- Plus the detectors that are not statistics: a fitted RadioML-style CNN on raw I/Q, a PSD-CNN and a TF-CNN.

**Speaker notes:** Stress the word *independent*. Passing one test proves almost nothing -- a detector can always find another domain. The claim is that all eight pass simultaneously, and that a detector trained in the domains that matter still only reaches 55.9 percent. Also be ready for 'why equal weights?' -- answer: it is a heuristic for comparison only, the paper Monte-Carlo'd the weights and the grade stayed Good or better in all 10^4 draws.

**For a 10-year-old:** Eight different lie-detector tests, all run at once. Passing one is easy. Passing all eight is the result.

### 19. Why it failed — and the v4 rewrite

*? min*


**Speaker notes:** Do not skip this act. It is the difference between a project that was assembled and a project that was engineered. Ten to fifteen minutes. It is also where you earn the room's trust, because you are showing your own failures before anyone can find them for you.

### 20. v1–v3: six silent bugs, each measured

*? min*


**Speaker notes:** Pick two of these and tell them properly, do not read the table. Bug 6 is the best story: a conjugate-mirrored preamble has *two* identical correlation peaks, so timing is ambiguous by exactly 32 samples, and BER sits at 50 percent with a perfectly healthy-looking loss curve. A week went into that. Bug 7 is the one academics care about most: our detector was scoring 80 percent because the test frames *were* the training frames. Fixing it moved the honest number to 55.9.

**For a 10-year-old:** Six times the system looked fine and was completely broken, and each time nothing on the screen told us. So we built tests that do tell us.

### 21. v4 design decisions, one by one

*? min*

- Framing: 512 payload + 64 keyed pilot = 576 samples = 2.3438 ms at 245760 S/s.
- 128 information bits per frame → 4 chips per bit → 9.03 dB processing gain.
- Preamble: Zadoff–Chu with an odd root derived from the session key — wrong key cannot lock.
- Receiver: matched filter first (it already works), the network only *refines* it.
- Channel, in the training graph: AWGN, CFO, phase noise, DC offset, IQ imbalance — randomised ±50 % around nominal, evaluated at nominal.
- One `LPIConfig` read by the trainer, the evaluator, the USRP scripts and the GRC blocks.
- >> The v1–v3 '512 vs 1024 frame' disaster was only possible because two files each had their own idea of the frame length.

**Speaker notes:** The single-source-of-truth point is a software-engineering point, and it lands well with technical experts. Say it as a rule you now enforce: no dimension exists in two files. Then give the number that makes it concrete: one config, four consumers.

**For a 10-year-old:** We used to keep the recipe in two notebooks, and they disagreed. Now there is one notebook.

### 22. Step 7 — Getting it onto the B210

*? min*

- `tx_usrp.py` / `rx_usrp.py`: raw UHD, no flowgraph needed. `uhd_io.py` is a version-tolerant wrapper.
- Sample rate 245760 S/s — 61.44 MHz / 250, an exact integer decimation, no fractional clock error.
- >> An old checklist said 32 kS/s. A B210 cannot go below ~208 kS/s. The stealth comes from the waveform, not from a slow clock.
- Carrier 2.484 GHz (ISM, Wi-Fi channel 14) — deliberately a band with real background activity to hide in.
- Two free-running local oscillators at 2.484 GHz are ~2 ppm apart ≈ ±5 kHz. Our pilot's fine sync is unambiguous only to ±213 Hz.
- >> So the receiver runs a two-pass coarse acquisition (±6 kHz at ~960 Hz bins by preamble peak ratio, then ±1 bin at ~30 Hz by decoder confidence). Verified: a ±2 kHz offset capture decodes to the exact text.
- TX gain is a link-budget knob, not a stealth knob — start at 0 dB, `--amp` 0.02–0.05.

**Speaker notes:** The ±213 Hz versus ±5 kHz story is the single best piece of RF reasoning in the project. Say it slowly: the pilot repeats once per frame, so per-frame phase only resolves the offset modulo fs/T_frame. Two independent crystals cannot be that close. Therefore a search is not optional -- it is mandatory, and it is built in so that on the bench 'wrong LO offset' is simply not a failure mode.

**For a 10-year-old:** Two radios are never tuned to exactly the same station. Ours can now find each other even when they are 5000 Hz apart.

### 23. Step 8 — The receive side

*? min*

- Sync chain: keyed-preamble correlation → timing → CFO derotation → per-frame phase from that frame's own pilot → matched filter → learned head → RS → AES → text.
- Ordering matters: derotate the CFO ramp *before* measuring pilot phase, or you double-count 2π·Δf·t_pilot. Sync looks perfect, BER is 0.5.
- Every capture prints: `frames`, `offset`, `CFO`, `coarse`, `pilot-SNR`, `dropped`, `crc N/N`.
- >> If it prints NO LOCK it is one of four things — frequency, sample rate, gain, key. Those four fields say which.
- Two failure modes we had to make distinguishable: frame-grid *phase* vs first active frame; and dropped frames vs a saturated model. Both now print on one line.

**Speaker notes:** For the demo this slide is your troubleshooting script. If the live link fails, you will look at exactly these four fields in front of an audience -- so say now, calmly, that the tool tells you the cause and you have a card for it. That is what a professional looks like under failure.

**For a 10-year-old:** When it does not work, the screen tells us which of four things is wrong. We do not guess.

### 24. Step 9 — The web console, wired to the real stack

*? min*

- The page had four defects: it could not even import (`from Crypto.Util.Counter import Counter`), it looked for a `templates/` folder that did not exist, it loaded the legacy v1 model, and `/api/metrics` returned hard-coded constants (KS 0.91, adversary 52.3 %, BER 0.3 %).
- >> A panel that displays numbers nobody measured is worse than no panel. So it now reads `lpi_v4/run/eval_arena.json` — the file `lpi_eval.py` writes — or says 'not measured'.
- `/api/send` now runs the real v4 chain: FrameCodec → build_iq → 1728 samples / 7.03 ms of IQ, and shows you the ciphertext hex.
- `/api/receive` runs the real decoder through the paper's 5-impairment channel at a settable SNR — or `--uhd` for a real radio.
- `/api/spectrum` is a real Welch PSD of the waveform generated in this session, not random numbers.
- Verified this session: 'ALPHA-INDIA-001' → encrypt → waveform → channel @ 6 dB → decode → CRC 1/1 OK, exact text, 0.02 s.

**Speaker notes:** Tell this one straight. The web page in the repository was a mock-up and it was broken. We found that, and we fixed it, and the fix was to make every number on it come from the same measurement file the paper quotes. If an examiner opens the Metrics tab they will see KS 0.865 and adversary 55.9 percent with the source filename and the protocol printed above it. Old files are kept as app.py.legacy-v1.bak / index.html.legacy-v1.bak.

**For a 10-year-old:** The dashboard used to have fake numbers painted on it. Now every number on it is one we actually measured.

### 25. Step 10 — Making the numbers trustworthy

*? min*

- `pytest tests -m 'not slow'` — 20 passed, 2 deselected (verified today, 2.4 s).
- `check_docs.py` — 82 documented commands must name real scripts and real flags (verified today: OK).
- CI on ubuntu + windows; nightly retrains 40 epochs and compares to `metrics/last.json`, opening an issue only on real regression.
- `export_manifest.json` records the sha256 of the checkpoint each TorchScript came from — because we once shipped an export that was not the model we measured (28 of 28 decoder tensors differed).
- A test AST-diffs every `args.<name>` against the argparse dests — a flag used but never defined gave two different exit codes and cost an afternoon.
- >> Clopper–Pearson one-sided 95 % upper bounds on BER and FER, so '0 %' is reported as what the sample actually excluded.

**Speaker notes:** This is the slide that separates you from every other student project. The specifics matter: name the sha256 manifest, name the AST diff, name the bounds. Then make the argument: a gate that measures the training budget rather than the code is worse than no gate -- CI used to assert KS > 0.05 on a 6-epoch model and went green and red on consecutive commits with no functional change.

**For a 10-year-old:** We built alarms that ring when a number stops being true — even if the program still runs.

### 26. Proof

*? min*


**Speaker notes:** Twenty minutes. Then the Q&A will live here.

### 27. Measured results — shipped checkpoint, one command

*? min*


**Speaker notes:** Read three rows aloud, not all twelve: KS, adversary, FER. Then point at the last column of the adversary row and volunteer the caveat yourself -- 55.90 percent at the paper's detector budget, 58.80 percent if you give the detector twice the training steps. That is the sentence that makes an examiner believe every other number on the slide.

**For a 10-year-old:** Twelve tests. Twelve passes. And we tell you the one test that gets slightly worse if the enemy tries harder.

### 28. BER and FER against SNR — with honest bounds

*? min*


**Speaker notes:** Explain `--n-msg 64` as arithmetic, not a style choice. With 4 texts a clean run only bounds FER at 52.7 percent. 16 texts bounds it at 17.1, 32 at 8.9 -- still short of the 5 percent target. 64 texts with none lost bounds it at 4.6 percent, which clears the line. So 64 is the minimum sample that can support the claim, and `cp_n_needed()` computes it in reverse: 59 texts for the FER line, 299 bits for the BER line. Also say plainly that 0 dB is below the claimed operating point -- the gate is taken over SNR ≥ 5 dB and the full sweep is printed either way.

**For a 10-year-old:** Saying 'zero mistakes' with only four tries proves nothing. We did 64 tries so the statement means something.

### 29. Prior work — the four-way intersection

*? min*


**Speaker notes:** Do not over-claim here. Say: the comparison is against what the surveyed papers report, not against systems we could test. Our claim is narrow and specific -- the simultaneous demonstration of four things. If someone says 'the autoencoder work is more elegant', agree: it is, and it is also unencrypted and validated on zero EW domains.

**For a 10-year-old:** Everyone else did one or two of these things. We are the ones who did all four together and flew it.

### 30. How the enemy was tested — and the trap we fell into

*? min*

- Two conventions change the answer by 25–30 points, and both were wrong at first.
- >> 1. A warden may swap its own output labels for free. 'Accuracy 3 %' for a detector that is 97 % accurate inverted is a lie. Every number we report is max(acc, 100−acc), and AUC is folded the same way.
- >> 2. The two classes must be disjoint pools. Our first fit seeded train and test from the same generator seed — the test frames *were* the training frames. The CNN reported 80–82 %, AUC 0.90. After the fix: 55.9 % / AUC 0.586.
- The detector is re-fit from scratch for the measurement: 2600 frames, 400 steps, Adam 1e-4, β = (0.5, 0.999). `run/warden_fitted.pt` is saved so a reviewer can re-score the exact detector.
- >> We also measured what does *not* work: raising adversarial pressure. `--w-adv 1.5` drove KS p from 0.87 down to 0.10–0.30 and left the warden at 65–72 %. Over-fitting one discriminator warps the samples without buying stealth.
- Few-shot: the enemy needs up to n = 2000 samples and still sits at 37.5–52.1 %, while BPSK hits 100 % at n = 10.

**Speaker notes:** This is where you convert sceptics. You are showing that you understand your own metric well enough to know the three ways it can be faked, and that you closed all three. The dead-end experiment is the strongest part: you tried the obvious improvement, it made things worse, and you wrote it down so nobody repeats it.

**For a 10-year-old:** We cheated by accident at first — we tested the robot on questions it had already seen. When we stopped, the honest score was worse. We report the honest score.

### 31. What we have NOT proved — stated before you ask

*? min*

- Live UHD streaming at 245760 S/s with `tx_usrp.py`/`rx_usrp.py` on hardware — written, and the offline equivalents pass, but the numbers in the table are simulation.
- PA-on operation and an antenna-separation / range sweep — not yet done.
- Multipath and frequency-selective fading: the channel model has 5 impairments, no real urban / jungle / high-altitude propagation.
- Single fixed carrier — no cognitive frequency selection, so a narrowband enemy who knows 2.484 GHz can sit on it.
- >> LO leakage is a real LPD vulnerability: the USRP's transmit LO leaks a spectral line exactly at the carrier offset. Our own receiver uses that line to estimate CFO — which means an eavesdropper can see it too.
- Payload is 24 characters per 7.03 ms burst. No voice, no streaming, no multi-frame reassembly protocol yet.
- >> One sync per short capture decodes only 68–72 % of the time — flat in burst count and in SNR from 6 to 10 dB. Cause, measured: the CFO slope is least-squares fitted over just `n_frames` pilot points, so on a 3-frame burst it can land in the wrong de-alias branch and acquire on the wrong sample. The `0 % FER` in our benchmark is a 64-message, one-sync protocol number.
- >> The shipped fix is ARQ: repeat the RS codeword and retransmit until a burst passes CRC. Measured through the console's own HTTP API — 40/40 exact round-trips at 6 dB and 40/40 at 10 dB, in 1–4 attempts (mean 2.4–2.6). Say it as '99 %+ per message with ARQ over a ~70 % single-capture link'.

**Speaker notes:** Read the LO-leakage bullet carefully and do not soften it. It is the most sophisticated criticism available to an RF examiner and you are making it first. Then give the mitigations: minimum TX gain, a well-calibrated board, and a hardware IQ/DC calibration stage. The per-capture 68–72 % figure is the most important honest number in the whole deck: it is the one an RF examiner could find by running your own console ten times, so say it first and say the ARQ fix in the same breath. Two earlier explanations for it were tested and retracted — `cfo_max` (every measured CFO was ≤197.7 Hz), the random latent seed, and zero-padding. If an earlier slide or answer implied `cfo_max`, correct it here.

**For a 10-year-old:** Here are the things we do not know yet. We would rather tell you now than have you find them.

### 32. The Indian Army use case

*? min*


**Speaker notes:** Twenty minutes. This is the act where officers engage. Be concrete about integration points and be equally concrete about the certification and logistics obstacles.

### 33. Where this fits in a real operation

*? min*

- >> Patrol and section-level comms in an EM-contested zone: short encrypted text bursts with a footprint at the noise floor, so direction-finding gets nothing to triangulate.
- >> Radio-silence discipline: a unit under EMCON can still pass a 24-character status without announcing its position. Silence is currently the only safe option; this replaces it.
- >> UAS / drone datalink: covert telemetry and control that does not give away the ground-control station's grid square to a passive RF sensor or a CNN drone-signal classifier.
- >> Counter-UAS deception: the same generator, run with a payload that is not real, is an RF decoy emitter. Few-shot resistance is exactly the property you want — the adversary AI cannot learn your pattern from captured samples.
- Also: covert inter-unit coordination in counter-swarm ops; high-altitude and border-surveillance posts where every emission is logged by the other side.

**Speaker notes:** The Counter-UAS deception point is the one that gets attention, and it is honest: a generator that can make a signal indistinguishable from noise can also make a fake signal indistinguishable from a real one. Say it as a dual-use observation, not as a claim you have demonstrated it.

**For a 10-year-old:** Right now, if a soldier radios in, the enemy learns where he is. This lets him send 24 letters without anyone knowing he spoke.

### 34. Deployment: what it would take

*? min*

- Form factor A — two B210s + two laptops: what is on this table. Good for trials and training.
- Form factor B — Raspberry Pi 5 + B210 in a dry-box, phone connected to the Pi's hotspot: a man-portable terminal, roughly 15–20 W.
- Form factor C — rack or vehicle mount with an X310-class radio and a GPSDO for 10 MHz / PPS.
- >> Clocking: with a shared 10 MHz reference and PPS on both radios, pass `--clock-source external --lo-lock`. Without it, the automatic coarse search already covers ±6 kHz, which is far more than 2 ppm at 2.484 GHz.
- Key management is the real gap: today it is a passphrase typed at both ends, SHA-256-derived. Production needs a proper KDF, a rekey schedule, zeroisation and secure storage — and the same key re-keys the Zadoff–Chu preamble, so a wrong key is a total lock failure, which is a feature and an operational risk at once.
- Regulatory: 2.484 GHz is ISM and fine for trials. Anything operational needs an allocated band and a spectral mask.
- Ruggedisation, TEMPEST, and cryptographic certification are outside this project's scope — we are proposing a waveform and a validated link, not a certified radio.

**Speaker notes:** Do not promise a fielded system. Promise a validated waveform and a reproducible testbench, plus a clear list of what stands between this and a certified radio. Officers respect that distinction far more than enthusiasm. If asked 'could we use this next month', the answer is: as a training aid and a trials asset, yes; as an operational crypto, no, not until key management and certification are done.

**For a 10-year-old:** We have built the engine and proved it works. It still needs a body, a lock, and a government stamp.

### 35. Validation — how a board should judge this

*? min*

- Rung 1 — Unit: `pytest tests -m 'not slow'` → 20 passed. Framing, sync, crypto loopback, bit-order, export integrity.
- Rung 2 — Metrics gate: `lpi_eval.py` → exit code 0, 6/6 checks, tables auto-generated into LaTeX from the same run.
- Rung 3 — Bench without a radio: `rx_usrp.py --selftest` (TX → 5-impairment channel → RX → exact text) and `lpi_grc.py` (the exact blocks the flowgraphs embed).
- Rung 4 — Cable loopback: TX/RX port → ≥ 20 dB attenuator → RX2. Expect `frames > 0`, pilot-SNR > 12 dB, `crc N/N OK`.
- Rung 5 — Over the air, 1–3 m, TX gain 0 → 20 dB. Record `pilot_snr_db`, `cfo_hz`, `dropped_frames`, `crc_ok` against distance.
- >> Rung 6 — The experiment that actually matters: pilot-SNR versus distance is the measured LPI range. `rx_usrp.py --json-out rx_log.jsonl` gives one JSON line per capture, so a for-loop over gain and distance *is* the experiment.
- Rung 7 — Red team: let an independent team fit their own detector on captured I/Q. That number, not ours, is the operational one.

**Speaker notes:** Offer rung 7 out loud. Saying 'the number that matters is the one your own team measures, not mine' is the most credible thing you can say in a five-hour viva, and it is also true -- the 48–56 percent band is a property of the detector's budget.

**For a 10-year-old:** There are seven steps from 'it runs on my laptop' to 'trust it in a fight'. We have done five. Here are the last two.

### 36. Future work — ordered by value, not by ease

*? min*

- 1. Multi-frame payload with segmentation and reassembly — unlocks µ-law voice, which is the actual operational requirement.
- 2. Field trials beyond 100 m with real propagation: urban, jungle, mountain. Characterise BER, not just pilot-SNR.
- 3. Ensemble of wardens (5+) instead of one, and adaptive loss weighting against whichever is winning.
- 4. Cognitive frequency selection — sense the band and move, so a narrowband enemy cannot camp on a known carrier.
- 5. Full Berlekamp–Massey RS decoding for burst errors; the current codec is RS(42,34) with a 4-byte correction budget.
- 6. MIMO / spatial diversity for link margin without more power — which is the LPI-correct way to gain range.
- 7. Synchronisation without a detectable preamble at all — the Zadoff–Chu pilot is our remaining structured feature.
- >> And the one we would do first if we had a week: LO-leakage suppression and hardware IQ/DC calibration, because it is the cheapest honest reduction in detectability left.

**Speaker notes:** Item 7 is the intellectually interesting one and academics will like it: your preamble is a known structured signal, and a determined detector can correlate against a guessed Zadoff–Chu root. Keying the root mitigates it, but implicit synchronisation is the real fix.

**For a 10-year-old:** Next: let it carry a voice, test it in real hills, and remove the last thing that still looks like a signal — the sync burst.

### 37. The sentence to remember

*? min*


**Speaker notes:** Stop talking after this. Let it sit for two seconds, then say 'I am happy to take questions, and the hardware is live if you would like to try it yourself.'

**For a 10-year-old:** A lock hides your words. This hides the fact that you spoke.

### 38. Appendix

*? min*


**Speaker notes:** Only used on demand. Know where each one is.

### 39. A1 — The maths you may be asked to derive

*? min*

- Processing gain: G = 10·log10(2·frame_len / n_bits) = 10·log10(1024/128) = 9.03 dB.
- Frame timing: 576 samples / 245760 S/s = 2.3438 ms; one AES/RS burst = 3 frames = 7.03 ms; 128 bits / 2.3438 ms = 54.6 kb/s gross.
- Unambiguous CFO: the pilot repeats every total_len samples, so ±fs/(2·total_len) = ±245760/(2·576) = ±213.3 Hz.
- KS test: D = sup|F_n(x) − F(x)|; p > 0.05 means the Gaussian null cannot be rejected. It is a statement about the marginal only — which is why eight domains, not one.
- Clopper–Pearson, zero failures in n trials: the one-sided 95 % upper bound on the rate is 1 − 0.05^(1/n) ≈ −ln(0.05)/n. For 64 texts that is 4.6 %; for 65,536 bits the BER bound is 4.57e-5.
- ε-blend: x = √(1−ε)·s + √ε·n₀ with power normalisation, ε = 0.10 shipped. Measured SNR cost of the 0.30σ dither: 0.19 dB.

**Speaker notes:** Derive the first and the third on the whiteboard if asked. They take twenty seconds each and they show the numbers are not magic.

### 40. A2 — Glossary for a mixed room

*? min*

- I/Q — the two numbers that describe a radio wave at one instant: how far along, and how far around.
- AWGN — the unavoidable background hiss. Additive (it adds), White (all frequencies), Gaussian (bell-curve amplitudes).
- SNR — how much louder your signal is than the hiss, in decibels. 0 dB means equally loud.
- BER — the fraction of bits that arrived wrong.
- FER — the fraction of whole messages that did not survive. What a user actually experiences.
- Cyclostationary — a hidden rhythm. Digital signals repeat patterns; that repetition is detectable even when the signal is buried.
- KS p-value — the chance that a set of numbers this 'noise-like' came from something that is not noise.
- Processing gain — the reward for spreading one bit over many chips: 9 dB here, i.e. eight times the effective power.
- Zadoff–Chu — a sequence whose correlation with itself is a single sharp spike and with anything else is nearly zero. Perfect for 'am I there, and when?'

**Speaker notes:** If an officer asks a question using a term you have not defined, define it in one sentence and then answer. Never answer over an undefined term.

### 41. A3 — Where everything lives in the repo

*? min*

- `lpi_v4/lpi_core.py` — config, generator, decoder, warden, channel, sync. The only place a dimension exists.
- `lpi_v4/lpi_stats.py` — the 8-domain EW battery, the pass criteria, and the composite Sc.
- `lpi_v4/lpi_crypto.py` — AES-128-CTR/GCM + CRC-8 + RS(42,34) + framing. The only payload path.
- `lpi_v4/lpi_train.py` / `lpi_eval.py` — the trainer and the gate (exit 0 = all pass).
- `lpi_v4/tx_usrp.py` / `rx_usrp.py` / `uhd_io.py` — the radio. `lpi_grc.py` + `make_grc.py` — the GNU Radio blocks and flowgraphs.
- `app.py` + `templates/index.html` — the web console (legacy v1 kept as `*.legacy-v1.bak`).
- `tests/test_lpi_v4.py` — 20 tests including bit-order, export-sha256 and CLI-flag integrity. `lpi_v4/check_docs.py` — 82 documented commands.
- `FINDINGS.md` — the engineering record of every failure. `AUTOMATION.md` — Colab / CI / bench. `FIELD_TEST_CHECKLIST.md` — the one-pager.
- `lpi_v4/run/` — checkpoints, `eval.md`, `eval.json`, `lpi_v4.csv`, `export_manifest.json`.
- Legacy v1–v3: `models.py`, `train.py`, `test_metrics.py`, `tx_lpi_cgan.grc` — kept deliberately, they are the evidence for the failure story.

**Speaker notes:** Have FINDINGS.md open in a second window for the whole session. It is the most persuasive document in the repository.

### 42. A4 — Reproduce everything in four commands

*? min*

- `python -m pytest tests -q -m "not slow"`  → 20 passed, 2 deselected (2.4 s)
- `cd lpi_v4 && python check_docs.py`  → OK: 82 documented commands reference real flags
- `python lpi_grc.py`  → frames=72 cfo=−31.5 Hz pilot=24.4 dB text='SHADOWCOMM-v4-GRC-LOOPBA…' PASS
- `python lpi_eval.py --ckpt run/lpi_v4.best.pt --frames 512 --snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400`  → 6/6 PASS, exit 0 (~4 min)
- Training from scratch: `python lpi_train.py --epochs 45 --steps 24 --batch 48 --eps 0.10` — ~9 min on 2 CPU cores, ~25 min on a T4.
- Radio: `python tx_usrp.py --addr <ip> --freq 2.484e9 --rate 245760 --gain 0 --text "ALPHA-INDIA-001" --bursts 20 --loop --period 0.2` and `python rx_usrp.py --watch --gain 30 --json-out rx_log.jsonl`.

**Speaker notes:** If a technical expert says 'I want to run this myself', this is the slide. Four commands, no radio required, under five minutes, and the fourth one prints its own verdict.

### 43. Thank you

*? min*


**Speaker notes:** Then hand over the keyboard. The best thing that can happen in the next four hours is an examiner typing a message and watching it come back.

**For a 10-year-old:** Come and type something. We dare you to catch it.
