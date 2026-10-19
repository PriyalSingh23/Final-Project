# Master script — the five-hour run of show

A five-hour review is not a talk. It is **four acts of talking, one act of
hardware, and a very long conversation**. Plan for 40 % presenting and 60 %
questions. The deck is 139 minutes of stage time; the other three hours are
yours to spend on the panel's questions.

**The single most important structural decision:** demo first, explain second.
By minute five the panel has *seen* a flat spectrum and a decoded message.
Everything after that is answering the question they now have, instead of
building up to a payoff they have to take on faith.

---

## Timings

| Time | What | Deck | Notes |
|---|---|---|---|
| 0:00–0:05 | Cold open + live demo | 1–2 | No greetings longer than one sentence |
| 0:05–0:10 | The one-slide result + what this is | 3–4 | Land the six numbers |
| 0:10–0:20 | **Act 1** — how a radio gets caught | 5–8 | The "why" |
| 0:20–1:05 | **Act 2** — how we got here (9 steps) | 9–18 | The story. Do not rush |
| 1:05–1:15 | *Break — offer it, do not wait to be asked* | — | Panels need it |
| 1:15–1:45 | **Act 3** — failure and the v4 rewrite | 19–28 | Your credibility act |
| 1:45–2:15 | **Act 4** — proof, and what is unproven | 29–34 | Then open the floor |
| 2:15–3:15 | **Open Q&A, round one** | appendix on demand | See §Q&A discipline |
| 3:15–3:30 | *Break* | — | |
| 3:30–4:00 | **Act 5** — the Army use case | 35–40 | Officers re-engage here |
| 4:00–4:40 | **Hardware session** — hands over the keyboard | — | Let them type |
| 4:40–4:55 | **Q&A round two** + limitations re-stated | 33 | |
| 4:55–5:00 | Close | 41 | One sentence. Then stop |

If you are cut to 30 minutes: slides 1, 2, 3, 4, 15 (why not DSSS), 17 (AWGN
honesty), 19 (six bugs), 26 (results), 33 (limitations), 41. That is a complete
argument in ten slides.

---

## The word-for-word opening

Say this. It is short on purpose.

> "Good morning. Before I explain anything, I would like to show you something.
>
> On this table are two USRP B210 software-defined radios. In about ninety
> seconds I will send an encrypted message between them. While that happens I
> would like you to watch the spectrum display, because the interesting thing
> is what you will *not* see there.
>
> *(start RX, wait two seconds, start TX)*
>
> That is the transmitted signal's spectrum. No bump. No line. No shape. To a
> receiver that is not us, this is indistinguishable from the thermal noise of
> an empty room.
>
> *(let the decode print)*
>
> …and the message arrived, exactly. Those two facts — nothing on the spectrum,
> message received — are the whole project. Everything I say for the rest of
> the day is explaining how both can be true, and being precise about how much
> of it we have actually proved."

Then go to the results slide. Do not build up to it — put it on the screen
early and let it sit there as a promise you spend the day keeping.

---

## Act-by-act intent

### Act 1 — How a radio gets caught (10 min)

**Goal:** make the problem undeniable before you describe the solution.

The one line to say twice, once here and once in the close: **"Encryption
protects content. LPI protects existence."**

Land the kill chain (energy → DF → traffic analysis → AI classification →
jam/intercept) and the fact that steps one to three need nothing from the
cryptography. Then the BPSK n = 10 number: a detector reaches 100 % accuracy on
BPSK with ten training samples. That is the sentence that makes officers lean
forward and academics nod.

### Act 2 — How we got here (45 min)

**Goal:** show work, in the order it happened. This is the act examiners grade.

Tell it as a story with nine steps, and be concrete about each one — name the
file, show the flowgraph, give the number.

1. **Literature and the gap.** Three pillars (end-to-end deep learning,
   conditional GANs, the √n covert law), and the four-way intersection nobody
   had demonstrated.
2. **GNU Radio, the first transmission.** Start with a tone and a QT sink. Then
   a real USRP source and sink. Tell them about the first time a peak that
   existed only as a number in a Python block appeared on a spectrum display on
   another machine.
3. **The test bench.** `TESTbench.grc`: file source → unpack k bits → chunks to
   symbols → RRC filter → GLFSR spreading → USRP. *Say why this matters:* you
   built a conventional spread-spectrum link **before** touching a neural
   network, so you know what a good link looks like.
4. **The payload layer.** AES-128-CTR, CRC-8, RS(42,34), 24 characters → 336
   coded bits → 3 frames. Tell the three bugs: the 336→256 truncation that
   destroys 80 payload bits, TX and RX speaking different protocols, and the
   fixed AES-CTR counter.
5. **The C-GAN.** Three networks, three jobs. The key insight to say slowly:
   *the generator is a learned keyed spreading codebook*, so every sample is
   the sum of 128 independent keyed chips — and a sum of many independent
   things is Gaussian.
6. **The nine-component loss.** One term per detector. Power normalisation
   weighs most because a radiometer is a diode and an integrator.
7. **Why not DSSS/FHSS/QPSK/OFDM.** Measured, not asserted. Every classical
   family hits 100 % detector accuracy; ours is 55.6 %.
8. **How we hit the AWGN statistics.** The honest version: an untrained
   generator already passes 8/8. The stealth is the central limit theorem; the
   learning buys invertibility and kills the covariance fingerprint.
9. **The eight-domain evaluation.** Independent referees. Passing one is easy;
   passing all eight is the result.

### Act 3 — Failure and the rewrite (30 min)

**Goal:** earn trust by showing your own failures first.

Do not skip this. Pick **two** of the six bugs and tell them properly rather
than reading the table. The best two:

- **The conjugate-mirrored preamble.** Two identical correlation peaks → 32-sample
  timing ambiguity → BER pinned at 50 % forever, with a healthy loss curve and
  every stealth metric green. Cost a week.
- **The warden that saw its own training data.** 80–82 % accuracy, AUC 0.90 —
  because the test frames *were* the training frames. Honest number: 55.9 %.

Then the v4 design rules: one config, four consumers; matched filter first,
network refines; the channel is in the training graph; phase is a channel
quantity, not a generator output.

Finish with the ±213 Hz versus ±5 kHz story — it is the best piece of RF
reasoning in the project and technical experts will recognise it immediately.

### Act 4 — Proof (30 min, then open the floor)

**Goal:** make the numbers reproducible in the room's head, not just visible.

Read **three** rows of the results table aloud, not twelve: KS, adversary, FER.
Then volunteer the caveat on the adversary number yourself — 55.9 % at 400 fit
steps, 58.8 % at 800. Saying it first is worth more than any other sentence in
the act.

Explain `--n-msg 64` as arithmetic: 4 texts bounds FER at 52.7 %, 16 at 17.1 %,
32 at 8.9 %, 64 at 4.6 %. That is why 64 is the minimum sample that supports
the claim.

Then the limitations slide, *before* anyone asks. Read the LO-leakage bullet
carefully and do not soften it: your own receiver uses the leaked LO line to
estimate CFO, so an eavesdropper can see it too.

Close the act by offering the red team: "The number that matters is the one
your own detector measures, not mine."

### Q&A round one (60 min)

See §Q&A discipline below.

### Act 5 — The Army use case (30 min)

**Goal:** convert a research result into an operational proposition, without
over-claiming.

Four concrete scenarios, not a wish list: patrol comms in a contested zone,
EMCON discipline, UAS datalink, counter-UAS deception. Then the three form
factors, the clocking story, and — critically — what stands between this and a
fielded radio: key management, anti-replay, PA characterisation, propagation
trials, certification.

*The sentence officers respect most:* "As a trials and training asset, yes. As
an operational crypto, not until key management and certification are done."

Then validation as a seven-rung ladder, ending with the red team.

### The hardware session (40 min)

Hand over the keyboard. The best possible outcome of the day is an examiner
typing a message and watching it come back.

Sequence:
1. Let them type a message in the web console. Send. Show the ciphertext hex.
2. Change the session key on one side only. Send. Show `NO LOCK` — explain that
   this is the LPE argument working as designed.
3. Open `rx_log.jsonl` and show the per-capture JSON: pilot SNR, CFO, CRC.
   "This is the raw data behind every claim."
4. Offer the range sweep if time allows (§7 of the runbook).

### Close (5 min)

One slide, one sentence:

> "Encryption protects content. LPI protects existence. This project adds the
> second one to a working, encrypted, error-corrected link — and proves it with
> eight independent statistical domains, a trained detector at 55.9 percent,
> and a reproducible hardware path.
>
> What we have not proved, we have listed. The hardware is live — please try
> it, and please try to break it."

Then stop talking. Silence for two seconds is not awkward; it is how a talk
ends.

---

## Q&A discipline

**Structure every answer the same way:** one-sentence direct answer → the
evidence → the honest limit. Then stop. Long answers manufacture follow-ups.

**Four phrases that save you:**

- *"That is a real limitation, and here is what we have done about it…"*
- *"We measured that, and the answer surprised us…"*
- *"I don't know, and here is how I would find out…"*
- *"The honest answer is that the number belongs to the detector, not to us."*

**Never:**
- Answer a question you were not asked.
- Quote a number you cannot source. Every number in the deck is in
  `09-METRICS-CARD.md` with its origin.
- Say "obviously" or "as everyone knows". It insults the room and marks you.
- Argue with an officer about operations, or with a professor about statistics.
  Concede the framing, then give your measurement.

**If two examiners disagree with each other:** do not referee. Say *"that is a
genuine disagreement and our data does not settle it — here is the experiment
that would"*, and write the experiment on the whiteboard. You have just turned
a trap into a demonstration of method.

**If you are wrong:** say so immediately, in one sentence, and move on.
*"You are right — that claim is not supported by what we measured. Thank you."*
Nothing recovers a room faster, and nothing destroys one slower than defending
an error for three minutes.

---

## Psychology notes — what actually happens in a five-hour review

- **Peak–end rule.** People remember the high point and the ending. Your peak
  is the opening demo; your end is the one-sentence close. Do not spend either
  on logistics.
- **Attention is cyclical, roughly 15–20 minutes.** Never let two consecutive
  20-minute blocks be the same *modality*. Alternate: slides → hardware →
  whiteboard → slides → conversation. Act 3 is where you must change modality,
  because that is where attention dies.
- **Cognitive load.** One idea per slide, one number per claim. The panel is
  holding a threat model, a signal chain, a loss function and a results table
  at once. Every extra ornament costs you comprehension elsewhere.
- **The credibility asymmetry.** Volunteering a limitation costs you nothing
  and buys a great deal. Being caught hiding one costs everything. You have
  three volunteered weaknesses in the deck (adversary budget, LO leakage,
  unproven PA) — make sure all three are said by you, out loud.
- **The whiteboard is your best prop.** Deriving 9.03 dB or ±213 Hz live, in
  twenty seconds, convinces a technical expert more than any slide. Practise
  both.
- **Answer to the room, not to the asker.** Repeat the question in your own
  words first — it buys thinking time, it makes sure you understood it, and it
  lets the rest of the panel follow.
- **Keep `FINDINGS.md` open all day.** When someone asks "did anything go
  wrong?", alt-tabbing to a document that lists your own failures in detail is
  the most persuasive move available to you.
