# PROJECT ANSHUMAN — Presentation pack

Built 2026-10-18 for the five-hour review. Everything here was generated from
this repository and the two reference papers; every number has a source in
`09-METRICS-CARD.md`.

## What is in this folder

| File | What it is | When to use it |
|---|---|---|
| `ANSHUMAN_deck.html` | The slide deck. Self-contained, no internet, opens from a file. 43 slides, ~139 min of stage time. | Project this. |
| `ANSHUMAN_deck.pptx` | The same deck as an editable PowerPoint, speaker notes embedded on every slide. | Use if the venue insists on PowerPoint, or to edit wording. |
| `00-README.md` | This file. | — |
| `01-MASTER-SCRIPT.md` | The five-hour run of show: timings, transitions, the word-for-word opening and closing, and what to do when the hardware fails. | Read tonight. Rehearse twice. |
| `02-SLIDE-BY-SLIDE.md` | What each slide is for, what to say, what to skip under time pressure. | During rehearsal. |
| `03-QA-BANK.md` | ~80 likely questions with answers, including the hostile ones and the ones you should say "I don't know" to. | The most important file in this folder. |
| `04-DEMO-RUNBOOK.md` | Exact commands, expected outputs, the table layout, and the failure decision tree. | Print this and keep it on the table. |
| `05-CHEAT-SHEET.md` | One page: every number, every formula, the four `NO LOCK` causes. | Print. Keep in your pocket. |
| `06-PRESENTATION-PATTERNS.md` | How the big labs and the best academics structure a technical talk, and what we copied. | Read once, tonight. |
| `09-METRICS-CARD.md` | Where every single number in the deck came from, and which ones we re-measured today. | When an examiner asks "where did that come from?" |
| `10-HANDOUT.md` | A one-page leave-behind for the panel. | Print ~10 copies. |
| `slides_data.py` | The single source of truth the deck is built from. | Edit this, then re-run the builders. |
| `build_deck.py` / `build_pptx.py` | Rebuild the HTML / PPTX after editing `slides_data.py`. | After any edit. |

Rebuild both after editing the content:

```bash
python presentation/build_deck.py
python presentation/build_pptx.py
```

## Deck controls

`→` or space = next · `←` = back · `N` = speaker notes · `G` = slide grid ·
`F` = full screen · `Home`/`End` = first/last · `0`–`9` = jump to that act.

## What changed in the repository today

The web console was broken — it could not even start. It is now wired to the
real v4 stack. Details and how to run it are in `04-DEMO-RUNBOOK.md` §5.
Old versions are kept as `app.py.legacy-v1.bak` and `index.html.legacy-v1.bak`.

Two findings from running the repository's own tools today are worth knowing
before you present (both are on the "what we have not proved" slide):

1. The shipped config carries `cfo_max = 300 Hz`, but the pilot's fine sync is
   unambiguous only to ±213.3 Hz. `lpi_eval.py` overrides it to 150 Hz so the
   benchmark is correct; `rx_usrp.py --selftest` uses the config value, which
   is why its 5 dB line can fail CRC. The field path is unaffected because
   coarse acquisition runs automatically. **This is a caveat, not the cause of
   anything — every CFO we measured was ≤197.7 Hz.** The real single-capture
   limitation is the few-point CFO slope fit, and ARQ is the shipped fix; see
   `09-METRICS-CARD.md` caveat 1 for the measured numbers.
2. `app.py` was using a fixed AES-CTR counter starting at 0 — keystream reuse,
   i.e. a two-time pad. That path is no longer reachable; the console now goes
   through `lpi_crypto.py`, which uses a fresh per-message nonce.
