# How the best technical talks are built — and what we copied

Read once, tonight. These are the patterns behind the deck's structure, so you
can defend the structure if anyone asks why it is not a conventional
"introduction → literature → methodology → results" dissertation layout.

---

## 1. What the big labs do

**Google / DeepMind research talks** open with the *problem*, not the method.
The first three minutes are a concrete failure that the audience already feels,
stated in domain language rather than jargon. The method arrives only after the
audience has agreed the problem is worth solving. One headline number per
slide, always with its error bar or confidence interval attached — a number
without uncertainty reads as marketing to a technical audience and they
discount the rest of the talk.

*What we copied:* the cold open (demo before explanation), and the fact that
every headline number in the deck carries its bound or its protocol.

**NVIDIA / systems-engineering keynotes** are benchmark-first and demo-live.
They put the measured comparison table on screen early, they show the hardware,
and — critically — they have a rehearsed fallback for when the live demo fails,
because live demos fail. The talk is designed so the demo is *evidence*, not
*entertainment*: if it fails, the argument still stands.

*What we copied:* the four-level fallback ladder in the runbook, and the rule
that the offline path exercises the identical code so the fallback is not a
different story.

**Apple-style product presentation** — one idea per slide, huge type, no
paragraphs, the speaker carries the content. It works because working memory
cannot hold a paragraph and a face at the same time.

*What we copied:* one claim per slide. The detail lives in the speaker notes
and in the written pack, not on the screen.

**Netflix / Spotify engineering blogs** lead with the failure. The most-read
postmortems are the ones titled "how we got it wrong". Failure-first
storytelling is not self-flagellation; it is the fastest route to being
believed when you later say something worked.

*What we copied:* Act 3 is a full act devoted to the six silent bugs, placed
*before* the results, not in an appendix.

---

## 2. What the best academics do

The strongest research presentations at IEEE/ACM venues share a shape:

1. **A gap sentence, stated as a negative.** "No prior work simultaneously
   demonstrates A, B, C and D." Specific, falsifiable, and it defines the
   contribution by exclusion so it cannot be dismissed as incremental.
2. **A threat/adversary model stated before the method.** Reviewers cannot
   judge a security claim without knowing what the adversary can do. Ours:
   passive, keyless, able to train supervised detectors on labelled samples
   under fixed preprocessing.
3. **Every claim paired with the protocol that produced it.** Frames, SNRs,
   detector budget, seeds. This is the single biggest difference between a
   credible and a non-credible empirical talk.
4. **Ablations and negative results.** "We tried X, it made Y worse, here is
   the measurement." This is what separates research from engineering theatre.
5. **Explicit limitations, in the author's own words, before the questions.**
6. **Reproducibility as a slide.** Not a footnote: the actual commands.

*What we copied:* all six. The deck has a gap slide, a threat model, a
protocol line on every results slide, two measured dead ends (`--w-adv 1.5`,
`--w-cov 3.0`), a volunteered-limitations slide, and an appendix slide with
four reproduction commands.

---

## 3. The psychology that actually decides how you are judged

**Peak–end rule.** People's memory of an experience is dominated by its
intensest moment and its ending, not its average. A five-hour review with a
brilliant middle and a tired ending is remembered as tired. Hence: the demo in
minute one, and a one-sentence close in the last minute with nothing
logistical after it.

**Cognitive load theory.** Working memory holds roughly four chunks. A slide
with six bullet points and a table asks for ten. Every unit of overload is
subtracted from the audience's ability to follow your *argument*, which is the
thing you are actually being graded on. Hence: one claim per slide, and the
detail in the notes.

**Attention cycles of 15–20 minutes.** Adult attention in a passive listening
mode drops sharply around the 15–20 minute mark and recovers on a change of
stimulus. Hence the deck alternates modality — slides, hardware, whiteboard,
conversation — and Act 3 (the failure story) sits at the first major trough,
because a story about things breaking is the most reliable way to recover a
room.

**The pratfall effect.** A competent speaker who admits a specific flaw is
rated *more* credible, not less. This only works if the flaw is specific and
self-identified. "There are some limitations" does nothing. "Our own receiver
uses the leaked LO line to estimate CFO, which means an eavesdropper can see it
too" does everything.

**Loss aversion in evaluation.** Examiners are more motivated to find what is
wrong than to confirm what is right. If you have already named the three
weaknesses, you have removed their leverage and redirected their energy toward
the substance. If you have hidden them, the discovery of the first one
recolors everything else you said.

**The Zeigarnik effect — open loops hold attention.** The opening demo creates
an unresolved question ("how can both of those be true?"). Do not close it
early. Let it stay open through Acts 1 and 2. That open loop is what keeps a
panel engaged through forty minutes of signal-chain detail.

**Social proof and institutional framing.** Cite the pillars you build on
(O'Shea & Hoydis, Mirza & Osindero, Bash et al.) by name and date. It signals
that you know the field, and it makes your novelty claim legible: the
contribution is the intersection, not any single component.

**Reciprocity.** Handing the panel something — a one-page summary, an offer to
run their own detector against your captures — creates an obligation to engage
in good faith rather than to attack. Hence the leave-behind handout and the
standing offer of a red-team test.

---

## 4. Delivery mechanics that are worth more than any slide

- **Stand still when making the key point.** Movement reads as uncertainty.
- **Pause two seconds before the number, not after.** "The adversary accuracy
  is … five five point nine percent." The pause before creates weight.
- **Repeat the question before answering.** Buys thinking time, confirms
  comprehension, includes the rest of the room.
- **Whiteboard the two derivations.** 9.03 dB and ±213 Hz. Twenty seconds each,
  and they convince a technical expert more than any slide, because they are
  verifiable in real time.
- **Never read a slide.** The panel can read faster than you can speak. If the
  slide says it, your job is to add the thing the slide cannot.
- **Manage your own energy.** Five hours is a marathon. Take the breaks you
  offer, drink water, and deliberately lower your volume in Act 4 so that the
  hardware session feels like a change of register.
- **Do not defend a lost point.** "You are right, that claim is not supported
  by what we measured. Thank you." One sentence. The room resets. Arguing for
  three minutes does not reset; it compounds.

---

## 5. Why this deck is not in dissertation order

A conventional dissertation layout — introduction, literature review,
methodology, results, conclusion — optimises for a *reader* who controls the
pace. A five-hour *review* optimises for a listener whose attention you do not
control.

So the deck is ordered by attention, not by convention:

| Conventional | This deck | Why |
|---|---|---|
| Introduction | Cold open + live demo | Creates the open loop that carries the next hour |
| — | Results, on slide 3 | A promise you then spend the day keeping; also inoculates against "where is this going?" |
| Literature review | Act 1: the problem, then the gap | The gap is only meaningful once the problem is felt |
| Methodology | Act 2: nine chronological steps | "How we got here" is a story; methodology sections are not |
| — | Act 3: failure, before results | Peak–end and pratfall; also the most engaging material at the attention trough |
| Results | Act 4: proof + limitations | Results are trusted because the failures came first |
| Conclusion | Act 5: operational use, then one sentence | Ends on what it means, not on what was done |

If an examiner asks why the structure is unusual, say exactly that: it is
ordered by attention rather than by convention, because a five-hour review is
a listening task, and here is the reasoning behind each placement.
