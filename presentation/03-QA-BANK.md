# Q&A bank — ~80 questions, with the answers

Read this file twice tonight. The answers are written to be *said*, not read:
short first sentence that lands, then the detail. If you only remember the
first sentence of each, you will still be fine.

Three rules that beat any single answer:

1. **Answer the question that was asked, then stop.** Long answers invite
   follow-ups you did not plan for.
2. **Volunteer the weakness before they find it.** "The honest limit here is X"
   buys more credibility than any number you have.
3. **"I don't know, and here is how I would find out"** is a strong answer.
   A bluff that gets caught in five hours costs you the whole room.

---

## A. The 30-second version (for the officer who says "explain it simply")

**Q. Tell me what this is, in plain words.**
> A radio that hides the fact that it is transmitting. Encryption stops the
> enemy reading your message; it does nothing about the enemy seeing that you
> spoke and finding out where you are. We make the transmitted signal
> statistically identical to background noise, and it still carries an
> encrypted, error-corrected message that comes out the other end exactly.

**Q. And a ten-year-old version?**
> Imagine 128 people each whispering one letter of your message at the same
> time, all in different voices. To anyone else it just sounds like a crowd.
> Your friend knows each of those voices, so she can pull every letter back
> out. The AI's job was to design 128 voices that a machine cannot tell apart
> from an actual crowd.

**Q. Why should I care?**
> Because right now a soldier has two options: transmit and be located, or stay
> silent. This gives him a third — send 24 characters without being located.
> That is the difference between a patrol reporting contact and a patrol that
> cannot report contact.

**Q. Is it deployed anywhere?**
> No. It is a validated laboratory prototype. What exists today is a waveform,
> a working link, a reproducible measurement method, and hardware on this
> table. What does not exist yet is key management, certification, and a
> ruggedised form factor. I would rather tell you that clearly than imply
> otherwise.

---

## B. Fundamentals — academics and non-specialists

**Q. What exactly is LPI, and how is it different from encryption?**
> Encryption is a payload-layer defence: it protects meaning. LPI is a
> physical-layer defence: it protects existence. An enemy does not need your
> plaintext — steps one to three of the intercept kill chain (energy detection,
> direction finding, traffic analysis) need nothing from your cryptography.
> We use both: AES-128-CTR plus CRC-8 plus Reed–Solomon for the payload, and
> the CGAN waveform for existence.

**Q. What is a "warden"?**
> Standard covert-communication terminology for the adversary receiver. In our
> threat model the warden is passive, has no keys, and can train supervised
> detectors on labelled examples under fixed preprocessing. That is a
> realistic worst case — it assumes the enemy has captured samples of similar
> traffic and knows what noise looks like in that band.

**Q. What does "KS p-value 0.865" actually mean?**
> The Kolmogorov–Smirnov test compares the empirical distribution of our I/Q
> amplitudes against a Gaussian. p = 0.865 means the null hypothesis "these
> samples came from a Gaussian" cannot be rejected — there is an 86.5 % chance
> of seeing a discrepancy at least this large from genuine noise. It is a
> statement about the *marginal* only, which is exactly why we also run seven
> other domains.

**Q. If the KS test passes, aren't you done?**
> No, and this is the most important caveat in the whole project. A low KS
> distance means the marginal power statistics match; it is completely
> insensitive to temporal correlation, phase structure and framing artefacts.
> A CNN can exploit any stable divergence the synthesis leaves behind. Low KS
> is necessary but nowhere near sufficient — that is why the headline number is
> the detector accuracy, not the KS test.

**Q. Why eight domains?**
> Because each detector family looks in a different place: distributional
> (KS, kurtosis), spectral (entropy), second-order (IQ circularity), envelope
> (PAPR), cyclostationary (SCF), higher-order (cumulant C42), and
> time-frequency (Wigner–Ville). Passing one proves almost nothing — an
> adversary can always switch domain. The claim is that all eight pass at once.

**Q. What is the composite score Sc and should I trust it?**
> It is a heuristic: the equal-weight mean of eight normalised deviations from
> the noise reference, where 0 is perfect. We report 0.140, which grades
> "Good" (Excellent is below 0.10). It is used for *comparison only* and never
> touches training. The paper Monte-Carlo'd the weights and the grade stayed
> Good or better in all 10⁴ draws, so it is not sensitive to the
> equal-weight assumption.

**Q. What is processing gain and where does yours come from?**
> Spreading one information bit over several chips lets the receiver integrate
> energy the way a detector cannot. Ours is 10·log₁₀(1024/128) = 9.03 dB —
> 128 bits carried on 4 real chips each. It is roughly eight times the
> effective power, and it is why the bit error rate is 3×10⁻⁵ at 6 dB instead
> of useless.

**Q. Why 2.484 GHz?**
> It is the ISM band, Wi-Fi channel 14 — licence-free for trials, and a band
> with real background activity, which is the environment an LPI waveform
> actually has to hide in. It is a testing choice, not an operational
> frequency plan; anything operational needs an allocated band and a spectral
> mask.

**Q. What is the data rate?**
> 128 information bits per 2.3438 ms frame = 54.6 kb/s gross at the frame
> layer. Net payload is 24 characters per 7.03 ms burst, because AES nonce,
> CRC and RS parity take the rest. That is deliberately a short-burst text
> link, not a streaming link — the square-root law of covert communication says
> covert throughput grows like √n, so broadband covert is not on offer from
> anyone.

---

## C. Why not the classical methods (you will be asked this)

**Q. Why not DSSS? It already looks like noise.**
> DSSS lowers the power spectral density, and it is genuinely good. But the
> spreading code is *periodic*, so a cyclostationary detector recovers the code
> rate and the chip structure. In our benchmark, at identical 512-sample frames
> and −5 dB SNR, a freshly trained binary detector separates DSSS from noise
> with 100 % balanced accuracy and KS p = 0.002. Our waveform sits at 55.6 %
> with KS p = 0.870.

**Q. Why not FHSS?**
> Same reason. The hop pattern is a periodic structure in the time–frequency
> plane — a spectrogram shows it directly. FHSS scored KS p = 0.001 and
> 100 % detector accuracy in the same benchmark.

**Q. Why not QPSK or BPSK?**
> BPSK is the clearest example of why this work exists. It only ever occupies
> two points on the complex plane, so its IQ circularity is 0.735 — against a
> noise reference of 0.055 and ours at 0.036. A detector reaches 100 % accuracy
> on BPSK with as few as ten training samples. QPSK is the same story with
> four points.

**Q. Why not OFDM?**
> The cyclic prefix is a deliberate, repeating correlation structure — it is a
> gift to a cyclostationary detector. OFDM: KS p = 0.507, 100 % detector
> accuracy.

**Q. Is your method just DSSS with extra steps?**
> Fair challenge, and partly yes — the core *is* a spreading codebook, and we
> say so plainly. The differences are: the codes are learned jointly with the
> receiver rather than drawn from a PN family; the waveform is shaped against
> a trained detector, including in the covariance domain; and the codebook is
> keyed, so the preamble will not even lock with a wrong key. Those three
> differences are what remove the signature that a cyclostationary detector
> lives on.

**Q. What about UWB?**
> Ultra-wideband lowers spectral density by spreading across enormous
> bandwidth. It works, and it is a legitimate LPI technique, but it costs
> bandwidth we do not have, it needs specialised hardware, and it is still
> detectable in the energy domain if the adversary integrates widely enough.
> It is a competitor, not something we beat — we did not benchmark it and I
> will not pretend otherwise.

**Q. Reed–Solomon or LDPC? Why RS?**
> RS(42,34) with 8 parity symbols corrects up to 4 bad bytes per codeword, is
> burst-error friendly, and decodes deterministically with a small,
> well-understood implementation — which matters when the codeword is 42 bytes
> and you need `nsize=42`, not the library default of 255. LDPC gives more
> coding gain but needs iterative decoding and is a poor fit for a 42-byte
> block. It is a listed future upgrade for a longer payload.

**Q. AES or ChaCha20?**
> AES-128-CTR is implemented and is what we validate; AES-GCM with a truncated
> 4-byte tag is also implemented for authenticated encryption. ChaCha20-Poly1305
> is a reasonable swap on constrained hardware and is not a research question —
> the interesting layer here is the waveform, and the crypto is deliberately
> standard so nobody has to trust our cryptography.

---

## D. The GAN and the training

**Q. What is a conditional GAN, in one sentence?**
> A generator that takes both random noise and a condition — here, the 128
> message bits — and produces output that both looks like real noise *and*
> encodes that specific condition, judged by a discriminator that is trying to
> catch it.

**Q. How is the generator actually built?**
> Its core is a learned keyed spreading codebook: a 128 → 1024 linear layer,
> one code vector per bit, so every output sample is the sum of 128 keyed
> chips. A small convolutional refinement head with gain 0.2 then removes
> residual structure. Latent input z is 64-dimensional. The decoder mirrors it:
> a matched filter against the same codebook, then a learned head producing one
> logit per bit.

**Q. Why a matched filter *and* a network? Isn't the network supposed to be the receiver?**
> That was the v1–v3 mistake. A local convolutional readout cannot invert a
> global code — that is a structural floor, not a training-length problem. So
> the matched filter runs first and already gives a working receiver; the
> network only refines it. That single change is a large part of why v4 works
> and v3 did not.

**Q. Are the weights the key?**
> Effectively yes. The session key derives the Zadoff–Chu root of the preamble
> and conditions the codebook. With a wrong key the receiver does not even get
> a correlation peak — it prints `NO LOCK`. That is a feature for LPE
> (exploitation resistance) and an operational risk at the same time, which is
> why key management is the biggest remaining gap.

**Q. Tell me the training configuration.**
> The paper runs: 500 epochs, batch 128, Adam with β = (0.5, 0.999), generator
> learning rate 2×10⁻⁴, discriminator 1×10⁻⁴, on an NVIDIA T4 in Colab. Nine
> independent runs across two messages and two carriers. The v4 stack in this
> repository reproduces the targets in 45 epochs at batch 48 — about 9 minutes
> on two CPU cores, 25 minutes on a T4.

**Q. What is the loss function?**
> Nine simultaneously active components: adversarial against D1 (weight 1.0),
> adversarial against a periodically reinitialised D2 (0.5), mean
> regularisation (0.5), variance regularisation (0.5), power constraint (3.0),
> spectral flatness (3.0), cyclostationary suppression (2.5), power
> normalisation (6.0 — highest, because energy radiometry is the cheapest and
> most sensitive EW tool), and diversity regularisation (3.0). v4 adds a
> covariance/Gram-spectrum term, `loss_cov`.

**Q. How did you choose the weights?**
> To keep the individual terms on comparable numerical scales during early
> training, then held fixed across all reported runs — deliberately *not* tuned
> post-hoc against any single stealth metric or detector outcome. That is a
> methodological choice, and it is stated in the paper so nobody can accuse us
> of fishing.

**Q. What are the three anti-fingerprinting defences?**
> One: stochastic Gaussian dither, σ_d = 0.30 σ̂ₓ, added at training and
> inference then renormalised — measured SNR cost 0.19 dB. Two: diversity
> regularisation, which maximises ‖G(z₁)−G(z₂)‖/‖z₁−z₂‖ so the generator cannot
> collapse onto one fingerprinted subspace. Three: D2's weights are
> reinitialised every 25 epochs, simulating an adversary who keeps retraining,
> so the generator cannot memorise a fixed decision boundary. Together they
> moved detection from 100 % to the mid-50s.

**Q. GANs are unstable. Did yours collapse?**
> Yes, in ways we documented. The discriminator can run away — we saw a +138
> discriminator loss drown every statistical term, which is why the objective
> is the bounded form (σ(D(x)) − 0.5)² with a deadband rather than a standard
> Wasserstein critic. And mode collapse is exactly what diversity
> regularisation exists to prevent.

**Q. Did raising the adversarial pressure help?**
> We measured it, and no — it is a documented dead end. A 30-epoch run at
> `--w-adv 1.5` with `--d-stop-frac 0.55` drove KS p from 0.87 down to
> 0.10–0.30 and left the warden at 65–72 %. Once the generator is optimised
> hard against *one particular* discriminator it distorts the samples to
> confuse that network, and the eight-domain battery punishes it. The
> statistical losses hold Gaussianity; the adversarial term only needs to stop
> the warden from winning.

**Q. What did the untrained-generator experiment show?**
> That the Gaussianity is largely free. With completely random weights, 512
> frames: KS p 0.857, kurtosis 2.998, entropy 0.959, circularity 0.041, 8 of 8
> domains pass, and 1.6×10⁻⁴ BER through the field path. The stealth comes from
> the central limit theorem over 128 keyed chips per frame plus the ε-blend.
> What the losses buy is keeping it Gaussian *while* the waveform is made
> invertible, and killing the covariance-domain fingerprint. Our advice to the
> next person: do not tune for KS.

**Q. Is that an admission that the AI is unnecessary?**
> No — it is a precise statement of what the AI is for. Without the learned
> refinement and the covariance term the waveform is noise-like but not
> reliably invertible, and it leaves a Gram-spectrum fingerprint that an
> eigenvalue detector can see. The CLT gives you invisibility for free; the
> learning gives you a working link that stays invisible.

**Q. What is `loss_cov`?**
> Any generator whose internal dimension is much smaller than the sample count
> produces frames living in a fixed low-rank subspace, so the batch Gram
> spectrum has outliers where i.i.d. Gaussian frames follow the
> Marchenko–Pastur law. That is precisely the statistic the eigenvalue detector
> family used in cooperative spectrum sensing exploits, and it is invisible in
> all eight per-frame domains. We penalise the log-spectrum mismatch on a 48×48
> Gram matrix. Reference values: 3×10⁻⁴ for AWGN vs AWGN, 0.285 for a rank-128
> signal vs AWGN.

**Q. Did doubling `loss_cov` help the adversary number?**
> No — also measured, also a dead end. Going from weight 2.0 to 3.0 bought
> about one point of adversary margin and cost the headline metric: KS p fell
> from 0.865 to 0.013, composite Sc doubled, and circularity failed. The
> promotion rule was fixed before measuring, and it said no.

---

## E. Security and cryptography

**Q. How strong is the encryption?**
> AES-128 in counter mode, key derived by SHA-256 from a passphrase, with a
> fresh 8-byte nonce per message, plus a CRC-8 over the framed data, plus
> RS(42,34). A GCM mode with a truncated 4-byte tag is implemented for
> authenticated encryption. The cryptography is deliberately standard and
> uninteresting — the contribution of this project is the layer underneath it.

**Q. Was it always that way?**
> No, and this is worth telling you. The earlier web-app path used a fixed
> counter starting at 0 for every message. Under AES-CTR, reusing a keystream
> is a two-time pad — it is a total-loss failure, not a weakening. That path is
> no longer reachable; the console now goes through `lpi_crypto.py`, where the
> nonce is per message. We found this by reading our own code critically.

**Q. What is your key management story?**
> Today: a passphrase typed at both ends. That is a lab answer, not a field
> answer. Production needs a real KDF, a rekey schedule, zeroisation and secure
> storage. There is a subtlety worth flagging: the same key also re-keys the
> Zadoff–Chu preamble, so a key mismatch is a total lock failure rather than a
> degraded link. That is good for security and bad for operations, and it
> needs an explicit key-sync protocol before this goes anywhere near a unit.

**Q. What about post-quantum?**
> AES-128 is not broken by Grover in any operationally relevant sense, and the
> field is moving to AES-256 as future-proofing. More important for us: the
> physical layer is not a post-quantum question at all. If an adversary records
> your I/Q now and attacks it in ten years, the question is whether they can
> detect that you transmitted — which is exactly what this project is about.

**Q. Replay attack?**
> The per-message nonce means a replayed burst does not decrypt to a new
> message, and in GCM mode the tag fails. But there is no anti-replay counter
> or timestamp in the framing yet — that is a real gap and it is on the future
> work list.

**Q. Side channels — the SDR leaks power.**
> Correct, and we found one ourselves: the USRP's transmit local oscillator
> leaks a spectral line exactly at the carrier offset. Our own receiver uses
> that line to estimate CFO — which means an eavesdropper can see it too. It
> is an LPD-level vulnerability, and the mitigations are minimum TX gain, a
> well-calibrated board, and hardware IQ/DC calibration. We discuss it in the
> write-up rather than hiding it.

**Q. Traffic analysis — you cannot hide that.**
> Largely true, and it is the honest limit of LPI. We shorten the exposure —
> 7.03 ms per burst — but burst timing, burst count and duty cycle are still
> observable by an adversary with enough integration time. A full answer needs
> cover traffic and scheduling discipline on top of this waveform. I would not
> claim otherwise.

---

## F. RF, hardware and the over-the-air story

**Q. Which SDR and why?**
> The papers validate on a BladeRF Micro 2.0 at 2.484 GHz / 2 MSPS, full
> duplex, real antennas, Windows transmit and Linux receive. This repository's
> v4 stack targets a USRP B210 at 245760 S/s, which is what is on this table.
> Both are commercial SDRs — the point is that this is not bespoke hardware.

**Q. Why 245760 S/s and not something slower for stealth?**
> Because a B210 cannot go below roughly 208 kS/s. An earlier checklist said
> 32 kS/s, and that is simply not reachable on this board. 245760 = 61.44 MHz /
> 250 is an exact integer decimation, so there is no fractional clock error.
> And the important point: the LPI margin comes from the waveform — 9.03 dB of
> processing gain and 128 bits per 2.34 ms frame — not from a slow sample rate.

**Q. Two radios never share a clock. How do you stay in sync?**
> That is the best question in RF land. Two free-running B210 oscillators at
> 2.484 GHz are about 2 ppm apart, which is ±5 kHz. But the pilot repeats once
> per frame, so per-frame phase only resolves the offset modulo fs/T_frame —
> ±213.3 Hz here. So a search is not optional, it is mandatory: a two-pass
> coarse acquisition searches ±6 kHz in ~960 Hz bins scored by preamble
> peak-to-median ratio, then ±1 bin at ~30 Hz scored by decoder confidence.
> Verified end to end — a ±2 kHz offset capture decodes to the exact text. If
> the two radios share a 10 MHz reference and PPS you pass
> `--clock-source external --lo-lock` and skip the search.

**Q. What RF impairments do you model?**
> Five, simultaneously: AWGN, carrier frequency offset, phase noise, DC offset,
> and IQ amplitude/phase imbalance. In training the parameters are drawn from
> ±50 % of the paper's nominal values; at evaluation the nominal values are
> used. That asymmetry matters — v1–v3 trained on a noiseless channel and the
> field receiver had nothing to stand on.

**Q. What did OTA actually show?**
> The Elsevier draft's OTA run: a 1619.7 ms capture at 2 MSPS (3,239,460
> samples), 2296 active windows out of 6327, frame sync by template
> cross-correlation gave 96 candidate peaks, 50 windows selected for majority
> vote. Per-window SNR mean 9.7 dB, max 21.5 dB. All 50 windows decoded the
> message `ALPHA_SECURE_IND_001A` correctly — 16,800 bits, BER 0.0000,
> confidence 1.0. The spectrogram across the full 2 MHz band is flat and
> featureless.

**Q. Isn't 50 windows a small sample for BER = 0?**
> Yes, and that is why we compute the bound rather than quoting the zero. With
> zero failures in 168,000 bits the one-sided 95 % Clopper–Pearson upper bound
> is about 1.8×10⁻⁵ per bit. We report "BER = 0.0000, 95 % upper bound
> 1.8×10⁻⁵" and not "perfect". The v4 numbers carry the same treatment.

**Q. TX gain — how do you set it for stealth?**
> TX gain is a link-budget knob, not a stealth knob. We start at 0 dB and an
> amplitude scale of 0.02–0.05 of full scale, then raise it only until the
> receiver locks. The stealth comes from the waveform's statistics, not from
> being quiet — although being quiet is free extra margin, and minimum gain is
> also the mitigation for LO leakage.

**Q. What about a power amplifier's non-linearity?**
> Not characterised yet, and it is a genuine open item. A saturated PA creates
> spectral regrowth and AM/AM-AM/PM distortion, both of which are exactly the
> kind of non-Gaussian artefact our eight domains are designed to catch — so
> the honest statement is that we do not know how much of our margin survives
> a real PA. That is the first field-trial experiment on the list.

**Q. What range have you achieved?**
> Bench-top only — indoor line of sight, 0.3 m antenna separation in the
> paper's hardware configuration. Free-space range, multipath and
> frequency-selective fading are all uncharacterised. The experiment that
> would answer this is already scripted: `rx_usrp.py --json-out rx_log.jsonl`
> writes one JSON line per capture with pilot SNR, CFO and CRC status, so a
> loop over gain and distance *is* the measured LPI range.

---

## G. Statistics, methodology, and honesty (the academic panel lives here)

**Q. How do you know your adversary number is not inflated?**
> Two conventions change it by 25–30 points, and both were wrong at first.
> First: a warden may swap its own output labels for free, so "3 % accuracy"
> for a detector that is 97 % accurate inverted is a lie. Every number we
> report is max(acc, 100 − acc) and AUC is folded the same way. Second: the two
> classes must come from disjoint pools. Our first fit seeded train and test
> from the same generator seed, so the test frames *were* the training frames
> and the CNN reported 80–82 % with AUC 0.90. After the fix — 2600 frames,
> 1300 train and 500 test — the honest number is 55.9 % with AUC 0.586.

**Q. Is 55.9 % actually good?**
> Chance is 50 %, so it is 5.9 points above chance, with AUC 0.583–0.586. That
> is "reduced detectability", not "no detectability", and the paper says so in
> those words. Operationally it means a dedicated classifier cannot be relied
> on to exploit us — but a determined adversary with more samples and more
> compute is not defeated by construction. We claim operationally challenging,
> not impossible.

**Q. You said the adversary number depends on the detector. Explain.**
> The same frozen generator reads 55.90 % (AUC 0.586) when the detector is
> fitted for 400 steps and 58.80 % (AUC 0.616) at 800 steps. So the 48–56 %
> band is met at the paper's detector budget and *missed* at twice it. We
> report both, with the fitting budget recorded next to the accuracy in the
> JSON. A number without its protocol is not reproducible, so the protocol
> travels with the number.

**Q. Few-shot — what does that curve show?**
> The detector is retrained from scratch at n ∈ {10, 25, 50, 100, 200, 500,
> 1000, 2000} samples per class, five seeds each, tested on a fixed 400-sample
> held-out set. Our waveform sits between 37.5 % and 52.1 % across the whole
> range with no upward trend; BPSK is at 100 % from n = 10. The absence of a
> monotonic trend is the real result — it is consistent with the theoretical
> prediction that a waveform statistically indistinguishable from AWGN presents
> no learnable discriminative feature.

**Q. Your BER is zero everywhere. That is suspicious.**
> It is suspicious, which is why we bound it instead of quoting it. 500 frames
> per SNR point from −10 to +10 dB, 168,000 bits per channel condition, zero
> failures, so the one-sided 95 % upper bound is about 1.8×10⁻⁵. And for the
> v4 stack the resolution problem is explicit: with 4 texts a clean run only
> bounds FER at 52.7 %, with 16 at 17.1 %, with 32 at 8.9 % — still short of
> the 5 % target — and with 64 texts at 4.6 %, which clears it. That is why the
> recommended command uses `--n-msg 64`. It is arithmetic, not a style choice.

**Q. At what SNR do you actually claim the link works?**
> The gate is taken over SNR ≥ 5 dB, stated on the command line as
> `--gate-snr-min 5`. The full sweep is printed either way: at 0 dB the field
> BER is 1.00×10⁻² and FER is 18.8 %, which fails, and 0 dB is below the
> claimed operating point. A gate that takes the worst over whatever SNR list
> you typed would be measuring the command line, not the link.

**Q. What is your null hypothesis in the EW tests?**
> An AWGN reference generated under matched preprocessing. Each of the eight
> tests compares our frames against that reference with its own pass criterion
> — for example |kurtosis − 3| < 0.15, entropy > 0.95, circularity < 0.15,
> SCF ratio < 3, |C42| < 0.3, WVD ratio < 3.7. Each test passes independently
> of the others; the composite is only a summary.

**Q. How do you defend against an adversary who uses the same eight tests?**
> They are welcome to. That is the point of reporting them: our waveform
> passes the tests an informed adversary would run, and the fitted CNN — which
> is not limited to those eight statistics — still only reaches 55.9 %. If
> someone finds a ninth domain that separates us, that is a genuine new result
> and we would want to know.

**Q. Is the warden's preprocessing fair?**
> Fixed and non-learnable by design: Welch PSD with a Hann window,
> nperseg 1024, noverlap 512, nfft 2048, FFT-shifted, dB, z-scored per example;
> and STFT with nperseg 256, noverlap 192, nfft 512. The threat model assumes
> the warden has fixed preprocessing and cannot learn it, which is the
> conventional assumption — and we state it so it can be challenged.

**Q. Nine seeds, two messages, two carriers. Is that enough?**
> For the detection claim, yes: ten independent detector seeds gave
> 55.6 % ± 1.2 %, 95 % CI [54.74 %, 56.46 %], t(9) = 14.76, p < 0.001, and
> repeating with 6400 samples instead of 3200 moved accuracy by ±0.8 %. For
> generalisation across environments, no — we have not tested real propagation,
> and that is the headline limitation.

---

## H. The failure story (asked by every good examiner)

**Q. What was the hardest bug?**
> A conjugate-mirrored preamble. A mirrored sequence has *two* identical
> correlation peaks, so timing is ambiguous by exactly 32 samples — BER sits at
> 50 % forever with a perfectly healthy-looking loss curve and every stealth
> metric green. It cost a week. The fix was a keyed Zadoff–Chu preamble with an
> odd root derived from the session key: measured aperiodic sidelobes 0.154, a
> single unambiguous peak, and a wrong key cannot lock at all.

**Q. What else went wrong?**
> Six root causes in total. No processing gain, so BER floored at 2.69 %
> regardless of SNR. MSE against ±1 labels, whose optimum sits *between* the
> two symbols, so soft outputs clustered at zero. I.i.d. dither with no
> whitening — stealth was perfect at KS p 0.99 while BER was 8 %. A noiseless
> training channel, so bench BER was 0.2 % and over-the-air was 0.5. Phase
> scrambled inside the generator, which a blind square loop cannot undo
> because of the ±90° ambiguity. And a warden trained on frames it was also
> tested on. Every one of these is written up in `FINDINGS.md` with the
> measurement that found it.

**Q. Any silent infrastructure bugs?**
> Several, and they are the ones I would tell a software engineer about.
> `np.correlate(a, conj(a))` conjugates its *second* argument, so it is not the
> autocorrelation — the Zadoff–Chu peak looked absent. `np.packbits` is
> MSB-first by default and so is GRC's `unpack_k_bits_bb`, so two conventions
> agreed by accident until someone made one explicit. `RSCodec(8)` defaults to
> `nsize=255`, so a 42-byte block silently returned `('', ok=True)` because an
> all-zero codeword has a valid CRC-8 of 0. And an exported TorchScript that
> was not the checkpoint we measured — 28 of 28 decoder tensors differed —
> which is now caught by a sha256 manifest and a test.

**Q. How do you stop the numbers rotting?**
> Three mechanisms. `pytest tests -m "not slow"` — 20 tests, including
> bit-order round-trips, export-sha256 identity, and an AST diff that checks
> every `args.<name>` against the argparse dests of every entry point.
> `check_docs.py` — 82 documented commands must name real scripts and real
> flags. And `lpi_eval.py` exits non-zero if any gate check fails, so CI can
> fail on a real regression.

**Q. Isn't a CI gate on an unconverged model meaningless?**
> Yes, and we learned that the hard way. CI used to assert KS > 0.05 on a
> 6-epoch smoke model; it went green on one commit and red on the next with no
> functional change, because the trainer gained one `torch.randn`-consuming
> export call that shifted which epoch became "best". A gate that measures the
> training budget rather than the code is worse than no gate. CI now asks only
> whether the receive chain works, and the nightly does the stealth regression
> against a converged model.

---

## I. Military and deployment

**Q. Where would this actually be used?**
> Four concrete places. Patrol and section-level comms in an EM-contested zone,
> where a short encrypted burst leaves nothing to triangulate. Radio-silence
> discipline — a unit under EMCON can pass a 24-character status instead of
> choosing between silence and exposure. UAS datalinks, so covert telemetry
> does not give away the ground-control station's grid square to a passive RF
> sensor or a CNN drone classifier. And counter-UAS deception: the same
> generator producing a payload that is not real is an RF decoy, and few-shot
> resistance is exactly the property you want because the adversary AI cannot
> learn the pattern from captured samples.

**Q. How would a unit integrate it?**
> Three form factors. Today's bench: two B210s and two laptops, good for trials
> and training. Man-portable: a Raspberry Pi 5 plus B210 in a dry box, phone on
> the Pi's hotspot, roughly 15–20 W. Vehicle or rack: an X310-class radio with
> a GPSDO for 10 MHz and PPS. In all three the software is the same —
> `tx_usrp.py` and `rx_usrp.py`, or the GNU Radio flowgraphs, or the web
> console.

**Q. Is it field deployable today?**
> As a trials and training asset, yes. As an operational cryptographic radio,
> no — and I want to be precise about what stands in the way. Key management is
> a passphrase on both ends. There is no anti-replay counter. The PA is not
> characterised. Propagation beyond bench-top line of sight is untested. And
> ruggedisation, TEMPEST and cryptographic certification are outside this
> project's scope entirely. We are offering a validated waveform and a
> reproducible testbench, not a certified radio.

**Q. What about jamming?**
> LPI and anti-jam are related but different goals, and we should not conflate
> them. The 9.03 dB of processing gain does buy jam margin — that is a real
> benefit. But an adversary who knows the band can blanket it, and we have a
> single fixed carrier with no cognitive frequency selection. Frequency agility
> is on the roadmap precisely because of this.

**Q. Can the enemy build the same thing?**
> Probably, in time — the techniques are published and we are publishing ours.
> The asymmetry we have is not the algorithm, it is the keying: the codebook
> and the Zadoff–Chu root are derived from a session key the enemy does not
> have, so even with an identical generator they cannot lock onto our traffic.
> That is an LPE argument, and it is why key management matters more than the
> network.

**Q. Does this violate any spectrum regulation?**
> 2.484 GHz is ISM, so trials are fine. Anything operational needs an allocated
> band and a spectral mask, and compliance is the operator's responsibility —
> that caveat is in the repository's own licence section.

**Q. Dual-use concern — this is an offensive tool.**
> It is a defensive tool with an obvious offensive application, and I will not
> pretend otherwise. The same generator that hides a friendly transmission can
> synthesise a convincing decoy. That is exactly why the work belongs in an
> institutional setting with an export-control-aware publication plan, and why
> the repository carries an explicit "educational and authorized research
> purposes only" disclaimer.

---

## J. Limits, future work, and "what would you do next"

**Q. What is the single biggest weakness?**
> Key management and the missing field trials, in that order. The waveform is
> the best-validated part of the system; the operational wrapper around it is
> the least developed. If I had a month I would spend it on a 100 m range
> sweep with real propagation and a proper key-sync protocol — not on making
> the network cleverer.

**Q. What is the biggest technical risk to the stealth claim?**
> The PA and the LO. Both create non-Gaussian artefacts that our eight domains
> are specifically designed to catch in simulation but that we have not
> measured on hardware with the amplifier on. Until that measurement exists,
> the honest statement is that the OTA stealth margin is unquantified.

**Q. What would you do with six more months?**
> In order of value: multi-frame payload with segmentation and reassembly,
> which unlocks µ-law voice — that is the actual operational requirement. Then
> field trials beyond 100 m in urban, jungle and mountain propagation. Then an
> ensemble of five or more wardens with adaptive loss weighting. Then
> cognitive frequency selection. Then a full Berlekamp–Massey RS decoder for
> burst errors, MIMO for margin without more power, and implicit
> synchronisation so there is no detectable preamble at all.

**Q. Why not voice now?**
> Because the payload is 24 characters per 7.03 ms burst and there is no
> segmentation-and-reassembly protocol. Voice needs a framing layer above the
> burst layer, a codec, and jitter handling. That is a protocol project, not a
> waveform project, and it would have delayed the result we actually have.

**Q. Is the preamble itself a giveaway?**
> Yes — it is the one remaining structured feature. The Zadoff–Chu root is
> keyed, so an adversary has to guess it, and a wrong root will not lock. But a
> determined adversary could correlate against a bank of candidate roots.
> Implicit synchronisation is the real fix and it is on the roadmap.

**Q. Would you scale the network up?**
> Not obviously. The current design is deliberately small — the stealth comes
> from the central limit theorem over the spreading, not from capacity. A
> bigger generator risks exactly the low-rank-subspace fingerprint that
> `loss_cov` exists to suppress. We would scale the *warden ensemble* before we
> scaled the generator.

**Q. Could this run in real time?**
> The generation is a single linear layer plus a small conv head over 512
> samples, so it is cheap. The untested part is the UHD streaming loop at
> 245760 S/s end to end — the scripts are written for it and their offline
> equivalents pass, but we have not measured sustained real-time throughput on
> the bench. That is the first thing tomorrow's hardware demo establishes.

---

## K. Questions you should NOT bluff

Say the first sentence, then stop. These are the ones where a guess is worse
than a gap.

- **"What is your measured range in metres?"** — "I don't have one. We are
  bench-top only; the sweep is scripted but not run."
- **"What is your BER with the PA on?"** — "Not measured. That is the first
  field experiment on the list."
- **"What is the detection rate against a state-level EW suite?"** — "Unknown.
  Our number is against a detector we fitted ourselves, at a stated budget. A
  red team with their own detector is the only honest answer, and I would
  welcome it."
- **"Has any of this been through a security evaluation?"** — "No. No
  cryptographic certification, no TEMPEST assessment."
- **"What does it cost per unit?"** — "I have not priced it. The bench on this
  table is two B210s, two laptops and two antennas; the man-portable version
  would be a Pi 5 and a B210, but I have not done a bill of materials."
- **"How does it compare to [specific classified system]?"** — "I cannot
  comment on systems I have no information about, and I would not want to
  speculate in front of people who do."

---

## L. If the hardware fails in front of them

1. Say out loud, calmly: *"The link did not lock. The receiver prints exactly
   four diagnostics — let me read them."* Then read `pilot-SNR`, `CFO`,
   `coarse`, `dropped`, `crc`.
2. `NO LOCK` has exactly four causes: **frequency, sample rate, gain, key**.
   Those four fields tell you which. Check them in that order.
3. If the pilot SNR is low: raise RX gain, reduce attenuation, confirm both
   ends are on 2.484e9 and 245760.
4. If it locks but CRC fails: check the session key matches on both ends — a
   wrong key costs everything, by design.
5. If it still fails after two minutes, switch to the prepared fallback
   (`04-DEMO-RUNBOOK.md` §6) and say: *"This is why the repository ships an
   offline path that exercises the identical code — let me show you the same
   chain without the radio, and I will come back to the RF fault."*
   Then come back to it. Do not abandon it silently.
