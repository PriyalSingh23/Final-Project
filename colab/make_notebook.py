#!/usr/bin/env python3
"""Build colab/LPI_v4_Colab_Training.ipynb from this file (single source of truth).

    python colab/make_notebook.py

Why a generator instead of a committed .ipynb: the notebook's commands must stay
byte-identical to what CI and the README tell you to run.  Editing one file keeps
them from drifting apart.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": ["\n".join(lines) + "\n"]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": ["\n".join(lines) + "\n"]}


CELLS = [
md("# ShadowComm LPI-CGAN v4 — Colab training + evaluation",
   "",
   "**What you get at the end of this notebook** (all saved to Google Drive):",
   "",
   "| file | what it is |",
   "|---|---|",
   "| `lpi/lpi_v4.pt`, `lpi/lpi_v4.best.pt` | full checkpoint (generator + decoder + wardens + metrics) |",
   "| `lpi/generator_lpi.pt`, `lpi/decoder_lpi.pt`, `lpi/lpi_config.json` | TorchScript + config for GNU Radio / the USRP scripts |",
   "| `lpi/lpi_v4.csv` | one row per epoch: every loss term and every gate metric |",
   "| `lpi/eval.md`, `lpi/eval.json` | the metric report (KS p, kurtosis, circularity, SCF, WVD, adversary, BER/FER) |",
   "",
   "Targets the last cell checks for you (the ones in `FIELD_TEST_CHECKLIST.md`):",
   "KS p > 0.05 (want > 0.85) · kurtosis ≈ 3.0 · adversary 48–56 % · BER < 1 % cable / < 5 % antenna ·"
   " cyclostationary ratio ≈ 1×.",
   "",
   "**Runtime → T4, and turn on *Settings → Run in background* if you start a long job** —"
   " the trainer saves every epoch, so a disconnect only costs the current epoch (`--resume` continues)."),

code("!nvidia-smi | head -15 || echo 'no GPU: the trainer works on CPU too (slower)'",
     "!pip -q install reedsolo pycryptodome pyyaml 2>&1 | tail -2"),

code("import os, torch, git  # git-python is preinstalled in Colab\n"
     "REPO = 'https://github.com/PriyalSingh23/Final-Project.git'\n"
     "BRANCH = 'main'\n"
     "if not os.path.isdir('/content/Final-Project/.git'):\n"
     "    git.Repo.clone_from(REPO, '/content/Final-Project', branch=BRANCH)\n"
     "repo = git.Repo('/content/Final-Project')\n"
     "repo.git.fetch('origin', BRANCH); repo.git.reset('--hard', 'origin/' + BRANCH)\n"
     "print('commit', repo.head.commit.hexsha[:8], '|', os.getcwd())\n"
     "os.chdir('/content/Final-Project/lpi_v4')\n"
     "import lpi_core, lpi_stats, lpi_crypto\n"
     "cfg = lpi_core.LPIConfig()\n"
     "g = lpi_core.LPIGenerator(cfg); d = lpi_core.LPIDecoder(cfg); d.tie_to(g)\n"
     "w = lpi_core.Warden(n_fft=cfg.frame_len, input='iq', ch=cfg.ch)\n"
     "n = lambda m: sum(p.numel() for p in m.parameters())\n"
     "print(f'frames {cfg.frame_len} samples / {cfg.n_bits} bits -> {cfg.gain_db:.2f} dB gain; "
     "frame {cfg.total_len} samples = {cfg.total_len/cfg.fs*1e3:.2f} ms')\n"
     "print(f'params: G {n(g)/1e6:.2f}M  Decoder {n(d)/1e6:.2f}M  Warden {n(w)/1e6:.2f}M')"),

code("from google.colab import drive\n"
     "drive.mount('/content/drive')\n"
     "RUN = '/content/drive/MyDrive/lpi'\n"
     "os.makedirs(RUN, exist_ok=True)\n"
     "print('checkpoints ->', RUN)"),

md("## Train\n",
   "",
   "`--epochs`/`--steps` are the two knobs that decide how long you sit here. The row printed each epoch is",
   "the whole gate, so you can stop as soon as it says `PASS` twice in a row:",
   "",
   "* `bce` — reconstruction loss on the *channel-corrupted* frame (this is what makes it decodable),",
   "* `link` — BER through the real field path (keyed preamble sync + CFO + phase noise + DC + IQ imbalance),",
   "* `adv` — balanced accuracy of the GAN's own warden (want 48–56 %),",
   "* `KS p / kurt / H / circ` — the noise-likeness battery, `EW n/8` — how many of the 8 domains pass.",
   "",
   "**The one knob that trades stealth against BER is `--eps`** (the masking-dither fraction ε, paper value 0.30σ):",
   "bigger ε → closer to Gaussian in every domain → higher BER. Start at 0.16, go up if `adv` or `KS` is the",
   "failing metric, down if `link`/`BER` is."),

code("EPOCHS, STEPS, BATCH = 60, 60, 96        # T4: ~25 min.  CPU: 24/48 -> ~40 min\n"
     "CMD = (f'PYTHONPATH=. python lpi_train.py --epochs {EPOCHS} --steps {STEPS} --batch {BATCH} '\n"
     "       '--eps 0.16 --w-cov 3.0 --w-adv 2.0 --d-stop-frac 0.55 '\n"
     "       f'--out {RUN}/lpi_v4.pt --log {RUN}/lpi_v4.csv --tb {RUN}/tb '\n"
     "       '--radioml --data ' + RUN + '/GOLD_XYZ_OSC.0001_1024.hdf5' )\n"
     "import shutil\n"
     "if not shutil.which('tensorboard'): CMD = CMD.replace(f' --tb {RUN}/tb','')\n"
     "print(CMD)\n"
     "# !{CMD} 2>&1 | tee {RUN}/train.log\n"
     "# ^ uncomment to run it.  Add `--resume` to continue from {RUN}/lpi_v4.pt after a disconnect."),

md("### Optional: sweep the one knob that matters",
   "",
   "Five short runs, each printing its own gate row; this is how the 0.16 default was chosen. Skip it if you are",
   "in a hurry — one `--eps 0.16` run is enough to pass the checklist on a good day."),

code("for eps in (0.10, 0.16, 0.22, 0.30):\n"
     "    !PYTHONPATH=. python lpi_train.py --epochs 12 --steps 24 --batch 64 --eps {eps} \\\n"
     "        --out {RUN}/sweep_eps{eps}.pt --log {RUN}/sweep_eps{eps}.csv --tb '' > /dev/null 2>&1\n"
     "    print(f'--- eps={eps} ---')\n"
     "    !tail -1 {RUN}/sweep_eps{eps}.csv"),

md("## Evaluate against the field gate",
   "",
   "`--fit-warden N` is the honest adversary protocol from the paper's Table 5: a *fresh* CNN detector is",
   "trained on 2 600 frames (70/15/15) and scored on held-out frames — the GAN's own discriminator is a moving",
   "target and is only reported as the pessimistic bound. Blocks: stealth, per-SNR BER, the AES/RS message layer,",
   "and a synthetic over-the-air capture decoded through the same `sync_and_correct` the USRP scripts use."),

code("CKPT = RUN + '/lpi_v4.best.pt'\n"
     "!ls -la {CKPT} || echo 'no checkpoint yet - run the cell above first'\n"
     "!PYTHONPATH=. python lpi_eval.py --ckpt {CKPT} --frames 512 --snrs 0,2,4,6,8,10 \\\n"
     "    --fit-warden 2000 --out-json {RUN}/eval.json --md {RUN}/eval.md\n"
     "print(open(RUN + '/eval.md').read())"),

md("## Export for the radio, and zip it to Drive",
   "",
   "Three files are all that the GNU Radio flowgraphs (`lpi_v4/tx_lpi_v4.grc`, `rx_lpi_v4.grc`) and the",
   "standalone USRP scripts need."),

code("import zipfile, glob, os\n"
     "# The trainer already wrote generator_lpi.pt / decoder_lpi.pt / lpi_config.json next to\n"
     "# every new best checkpoint, plus export_manifest.json naming the checkpoint (with its\n"
     "# sha256) those files mirror -- check it before wiring a GNU Radio block to them.\n"
     "# Re-export on demand with:\n"
     "!PYTHONPATH=. python -c \"import sys; sys.path.insert(0,'.'); from lpi_train import export_for_grc; export_for_grc('{CKPT}','cpu','{RUN}')\"\n"
     "files = sorted(glob.glob(RUN + '/*'))\n"
     "print('in Drive:', [os.path.basename(f) for f in files if os.path.isfile(f)])\n"
     "with zipfile.ZipFile('/content/lpi_v4_export.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
     "    for p in files:\n"
     "        if os.path.isfile(p) and p.endswith(('.pt', '.json', '.csv')):\n"
     "            z.write(p, os.path.basename(p))\n"
     "print('zipped ->', '/content/lpi_v4_export.zip', os.path.getsize('/content/lpi_v4_export.zip')//1024, 'kB')\n"
     "try:\n"
     "    from google.colab import files as cf\n"
     "    cf.download('/content/lpi_v4_export.zip')\n"
     "except Exception as e:\n"
     "    print('(browser download skipped:', e, ') - the files are also in Drive')"),

md("## Push the numbers back to GitHub (optional automation)",
   "",
   "Store a fine-grained PAT (scope: `repo` write) as the Colab **Secret** `GH_TOKEN`, then this cell commits",
   "`lpi_v4/run/eval.md` + the CSV to the repo so the field checklist and the paper tables are always generated",
   "by the run that produced the model — never typed by hand."),

code("import subprocess\n"
     "def sh(*c):\n"
     "    r = subprocess.run(c, cwd='/content/Final-Project', capture_output=True, text=True)\n"
     "    print(' '.join(c[:3]), '->', (r.stdout or r.stderr).strip()[:300])\n"
     "    return r\n"
     "# from google.colab import userdata; TOK = userdata.get('GH_TOKEN')\n"
     "# sh('git','remote','set-url','origin',f'https://x-access-token:{TOK}@github.com/PriyalSingh23/Final-Project.git')\n"
     "# sh('git','checkout','-B','main'); sh('git','pull','--rebase','origin','main')\n"
     "# for f in ('lpi_v4/run/lpi_v4.csv','lpi_v4/run/eval.md','lpi_v4/run/eval.json'):\n"
     "#     if os.path.exists('/content/drive/MyDrive/lpi/'+os.path.basename(f)):\n"
     "#         sh('cp','/content/drive/MyDrive/lpi/'+os.path.basename(f), f)\n"
     "# sh('git','add','-A'); sh('git','commit','-m','Colab run: metrics + checkpoint summary')\n"
     "# sh('git','push','origin','main')\n"
     "print('uncomment the lines above once GH_TOKEN is set as a Colab secret')"),

md("## Decode a field capture without leaving Colab",
   "",
   "Drop the `.npz` (written by `rx_usrp.py --save cap.npz`) into `MyDrive/lpi/` and run this — it is the exact",
   "same code path the receiver uses on the bench, so it predicts your antenna result."),

code("import glob\n"
     "caps = sorted(glob.glob(RUN + '/*.npz'))\n"
     "print('captures found:', caps)\n"
     "if caps:\n"
     "    !PYTHONPATH=/content/Final-Project/lpi_v4 python /content/Final-Project/lpi_v4/lpi_eval.py \\\n"
     "        --ckpt {CKPT} --capture {caps[-1]} --snrs 5 --frames 512\n"
     "else:\n"
     "    print('no capture yet: run  python rx_usrp.py --save {RUN}/cap.npz ...  on the laptop')"),

md("## If a metric will not move",
   "",
   "| symptom | first thing to change | why |",
   "|---|---|---|",
   "| `KS p < 0.05` | `--eps` up (0.22 → 0.30), `--w-stat` up to 2 | the marginal distribution is set by the dither blend; the statistical losses hold it there |",
   "| `adv > 56 %` | `--w-adv` up to 4, `--d-stop-frac` down to 0.4 | the warden has to be *fooled*, and it can only be fooled by a fixed target it is then pushed away from |",
   "| `link`/BER > 1 % | `--eps` down; check `bce` actually went down; more `--epochs` | dither buys stealth with BER, nothing else |",
   "| `cyclo / SCF > 1.5` | `--w-stat` up; longer frames (`--frame-len 1024`) | the cycle-feature penalty is a batch statistic; it needs enough samples per step |",
   "| kurtosis ≠ 3 | nothing — check `--eps` is not 0 | 128 keyed chips per frame are Gaussian by CLT; the dither only has to stop the codebook showing through |",
   "",
   "Do **not** re-add a per-frame phase scramble in the generator (the v3 attempt): the payload phase is a channel",
   "quantity, the pilot owns it, and scrambling the payload alone leaves the receiver on a ±90° blind ambiguity —",
   "the decoder then sits at 50 % BER forever with everything else looking healthy."),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"collapsed_sections": [],
                                                       "name": "LPI_v4_Colab_Training.ipynb",
                                                       "provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = os.path.join(HERE, "LPI_v4_Colab_Training.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, sum(len(json.dumps(c)) for c in nb["cells"]), "bytes of cells")


if __name__ == "__main__":
    main()
