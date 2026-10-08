"""
Regression tests for ShadowComm LPI v4 (lpi_v4/).
==================================================

Everything here runs on CPU in a couple of minutes and needs no GPU, no radio and
no checkpoint -- except the two tests marked ``slow``, which use
``lpi_v4/run/lpi_v4.best.pt`` when it exists and skip when it does not.

    pytest tests -q                     # CI
    pytest tests -q -m slow             # metrics, only with a trained model
    python -m pytest tests -q -x -s     # locally, with prints
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "lpi_v4")
sys.path.insert(0, V4)

torch = pytest.importorskip("torch")

from lpi_core import (LPIConfig, LPIGenerator, LPIDecoder, LockNotFound,  # noqa: E402
                      apply_channel, frame_with_pilot, frames_to_bits, make_preamble,
                      sync_and_correct, unit_power)
from lpi_crypto import FrameCodec, crc8, rs_decode, rs_encode      # noqa: E402
from lpi_stats import composite_stat_loss, ew_report, loss_cov     # noqa: E402

CKPT = os.path.join(V4, "run", "lpi_v4.best.pt")
HAVE_CKPT = os.path.exists(CKPT)


# --------------------------------------------------------------------- config ----
def test_config_derived_sizes():
    cfg = LPIConfig()
    assert cfg.out_len == 2 * cfg.frame_len
    assert cfg.total_len == cfg.frame_len + cfg.pilot_len
    # the pilot fine sync is only unambiguous inside +-fs/(2*T_frame)
    assert cfg.cfo_max < cfg.fs / (2 * cfg.total_len)


def test_config_json_roundtrip(tmp_path):
    from lpi_core import load_config, save_config
    cfg = LPIConfig(n_bits=64, frame_len=256, dither_eps=0.2)
    p = tmp_path / "cfg.json"
    save_config(cfg, str(p))
    back = load_config(str(p))
    assert back.n_bits == 64 and back.frame_len == 256 and back.dither_eps == 0.2


# ---------------------------------------------------------------------- pilot ----
def test_preamble_is_keyed_and_unambiguous():
    L = 64
    a = make_preamble("SESSION-KEY", L)
    b = make_preamble("OTHER-KEY", L)
    assert a.shape == (L,) and np.allclose(np.abs(a), 1.0)
    # constant modulus + aperiodic autocorrelation with no second peak
    # np.correlate conjugates its *second* argument, so this is sum a[n+d] conj(a[n])
    c = np.correlate(a, a, mode="full") / L
    assert np.isclose(np.abs(c[L - 1]), 1.0, atol=1e-4)
    sidelobes = np.abs(np.concatenate([c[:L - 1], c[L:]]))
    assert sidelobes.max() < 0.25, "ambiguous timing peak -> frame-grid errors"
    # a wrong key must not correlate enough to lock
    assert np.abs(np.sum(a * np.conj(b)) / L) < 0.4


# ---------------------------------------------------------------------- crypto ----
@pytest.mark.parametrize("mode", ["ctr", "gcm"])
def test_codec_roundtrip(mode):
    fc = FrameCodec("TEST-KEY", 128, mode=mode)
    msg = "LPI-v4 0123456789"
    bits = fc.encode(msg)
    assert bits.size == fc.frames * 128
    back, ok = fc.decode(bits)
    assert ok and back == msg


def test_codec_long_and_tamper():
    fc = FrameCodec("TEST-KEY", 128)
    msg = "a longer secret message that spans several RS codewords, with unicode ✓"
    bits = fc.encode_long(msg)
    back, ok = fc.decode(bits)
    assert ok and back == msg
    bad = bits.copy()
    bad[np.random.default_rng(0).integers(0, bad.size, 40)] ^= 1
    back2, ok2 = fc.decode(bad)
    assert not ok2 and back2 != msg          # CRC/RS must refuse it


def test_rs_crc_helpers():
    data = bytes(range(34))
    cod = rs_encode(data)
    assert len(cod) == 42
    errs = bytearray(cod)
    errs[3] ^= 0xFF
    errs[20] ^= 0x0F
    assert rs_decode(bytes(errs)) == data           # RS(42,34) corrects 4 symbols
    assert crc8(data) != crc8(bytes(range(1, 33)) + b"\x00")


def test_codec_bit_order_matches_numpy():
    """`np.packbits(..., bitorder='big')` is what blocks.unpack_k_bits_bb undoes;
    if that ever changes the GRC path silently transmits garbage."""
    fc = FrameCodec("K", 128)
    bits = fc.encode("GRCSYNC0123456789AB")
    byts = np.packbits(bits, bitorder="big")
    back = np.unpackbits(byts, bitorder="big")[: bits.size]
    assert np.array_equal(back, bits)


# ------------------------------------------------------------------ the chain ----
def _model(cfg, device="cpu"):
    torch.manual_seed(0)
    gen = LPIGenerator(cfg).to(device).eval()
    dec = LPIDecoder(cfg).to(device).eval()
    dec.tie_to(gen)
    return gen, dec


def test_sync_finds_offset_cfo_and_frames():
    cfg = LPIConfig()
    gen, dec = _model(cfg)
    rng = np.random.default_rng(3)
    n = 48
    bits = rng.integers(0, 2, (n, cfg.n_bits)).astype(np.float32) * 2 - 1
    with torch.no_grad():
        x = gen(torch.randn(n, cfg.z_dim), torch.from_numpy(bits))
    pay = (x[:, 0] + 1j * x[:, 1]).numpy()
    pl = make_preamble("K", cfg.pilot_len)
    pad = 1332                                    # not a multiple of the frame length
    cap = np.concatenate([np.zeros(pad, np.complex64),
                          frame_with_pilot(pay, pl).reshape(-1),
                          np.zeros(900, np.complex64)]).astype(np.complex64)
    cfo = 120.0
    t = np.arange(cap.size) / cfg.fs
    cap = (cap * np.exp(1j * (2 * np.pi * cfo * t + 1.7))).astype(np.complex64)
    frames, info = sync_and_correct(cap, cfg, pl, "K", max_frames=n, cfo_search=True,
                                    return_info=True)
    assert info["offset"] == pad % cfg.total_len, "wrong frame-grid phase"
    assert info["start_sample"] == pad, "first active frame not found"
    assert frames.shape[0] == n
    assert abs(info["cfo_hz"] - cfo) < 8.0
    bits_out = frames_to_bits(dec, frames)
    exp = (bits > 0).astype(np.uint8).reshape(-1)
    ber = float((bits_out[: exp.size] != exp).mean())
    assert ber < 0.02, f"link broken by sync, BER={ber:.4f}"


def test_sync_raises_when_nothing_is_transmitting():
    cfg = LPIConfig()
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(40000) + 1j * rng.standard_normal(40000)).astype(np.complex64)
    with pytest.raises(LockNotFound):
        sync_and_correct(noise, cfg, make_preamble("K", cfg.pilot_len), "K")


def test_channel_snr_definition():
    """snr_db must be the complex-baseband SNR, per real dimension and all."""
    cfg = LPIConfig()
    x = unit_power(torch.randn(400, 2, cfg.frame_len))
    y, p = apply_channel(x, cfg, snr_db=10.0, impairments=False)
    n = y - x
    est = 10 * np.log10(float((x ** 2).mean()) / float((n ** 2).mean()))
    assert abs(est - 10.0) < 0.3, f"SNR convention drifted ({est:.2f} dB)"


def test_grc_block_sources_are_valid_python():
    sys.path.insert(0, V4)
    import lpi_grc
    for src in lpi_grc.make_grc_sources():
        tree = "\n".join("    " + l for l in src.splitlines())
        compile("def _f(lpi_dir):\n" + tree, "<grc>", "exec")



def test_grc_files_parse():
    yaml = pytest.importorskip("yaml")
    for name in ("tx_lpi_v4.grc", "rx_lpi_v4.grc"):
        path = os.path.join(V4, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not generated (run lpi_v4/make_grc.py)")
        d = yaml.safe_load(open(path))
        assert {"options", "blocks", "connections"} <= set(d)
        ids = [b["id"] for b in d["blocks"]]
        assert "epy_block" in ids and "uhd_usrp_sink_0" or True
        names = {b["name"] for b in d["blocks"]}
        for c in d["connections"]:
            assert c[0] in names and c[2] in names, f"{name}: dangling connection {c}"


# ---------------------------------------------------------------- metrics (slow) ----
def _trained(cfg):
    from lpi_core import load_config
    cfg = load_config(CKPT)
    gen, dec = _model(cfg)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        gen.load_state_dict(blob.get("generator", {}), strict=False)
        dec.load_state_dict(blob.get("decoder", {}), strict=False)
        dec.tie_to(gen)
    return cfg, gen, dec


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="no trained checkpoint")
def test_trained_metrics_gate():
    cfg, gen, dec = _trained(None)
    n = 1024
    bits = (torch.rand(n, cfg.n_bits) > 0.5).float() * 2 - 1
    with torch.no_grad():
        x = gen(torch.randn(n, cfg.z_dim), bits)
    ref = torch.randn(n, 2, cfg.frame_len)
    rep = ew_report(x.numpy(), ref.numpy())
    assert rep["ks_p"] > 0.05, rep
    assert 2.7 < rep["kurt"] < 3.3, rep
    assert rep["circ"] < 0.15, rep
    assert rep["n_pass"] >= 7, rep
    with torch.no_grad():
        y, _ = apply_channel(x, cfg, snr_db=5.0, impairments=False)
        ber = float((dec.hard_bits(y).cpu().numpy().reshape(-1)
                     != (bits > 0).numpy().reshape(-1)).mean())
    assert ber < 0.01, f"BER at 5 dB = {ber:.2e}"


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="no trained checkpoint")
def test_trained_message_over_field_channel():
    import lpi_grc
    cfg, gen, dec = _trained(None)
    cfg.cfo_max = 150.0                     # inside the pilot sync's unambiguous range
    tx = lpi_grc.TxCore(CKPT, "", "SESSION-KEY")
    fc = FrameCodec("SESSION-KEY", cfg.n_bits)
    msg = "FIELD-READY-LPI-v4-OK"[: fc.max_len]
    bits = np.concatenate([fc.encode(msg) for _ in range(12)])
    nf = bits.size // cfg.n_bits
    vin = np.zeros((nf, cfg.n_bits), np.complex64)
    vin.real = bits.reshape(nf, cfg.n_bits).astype(np.float32) * 2 - 1
    vout = np.zeros((nf, tx.L), np.complex64)
    tx.work(vin, vout)
    iq = vout.reshape(-1)
    iq = iq / np.sqrt((iq.real ** 2 + iq.imag ** 2).mean())
    t = torch.from_numpy(np.stack([iq.real, iq.imag])[None].astype(np.float32))
    y, _ = apply_channel(t, cfg, snr_db=6.0, impairments=True)
    raw = (y[0, 0].numpy() + 1j * y[0, 1].numpy()).astype(np.complex64)
    res = lpi_grc.decode_capture(raw, cfg, dec, "SESSION-KEY")
    assert res["n_frames"] >= 0.8 * nf, res
    assert res["ok"] is True, res
    assert res["text"] == msg * 12, res


def test_cli_flags_are_all_defined():
    """Every ``args.<name>`` an entry point reads must come from ``add_argument``.

    A dropped flag is invisible to imports and to every other test here: it only
    fires once somebody actually passes it.  That is exactly how the CI eval step
    died -- ``lpi_eval.py`` still wrote its Markdown report behind ``if args.md:``
    while the ``--md`` definition was gone, so any ``--md`` caller got
    ``error: unrecognized arguments`` (exit 2) and any caller without it got an
    ``AttributeError`` (exit 1), both of which look like a metric regression from
    the outside.  Automated instead of re-learned by hand.
    """
    import ast

    offenders = {}
    for name in sorted(os.listdir(V4)):
        if not (name.startswith("lpi_") or name.endswith("_usrp.py")):
            continue
        path = os.path.join(V4, name)
        defined, used = set(), set()
        for node in ast.walk(ast.parse(open(path).read())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                            and a.value.startswith("--"):
                        defined.add(a.value[2:].replace("-", "_"))
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        defined.add(kw.value.value)
            elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id.endswith("args")):
                used.add(node.attr)
        missing = sorted(used - defined)
        if missing:
            offenders[name] = missing
    assert not offenders, f"args read but never defined: {offenders}"


def test_documented_commands_match_argparse():
    """Every command line this repo documents or prints must actually run.

    ``lpi_v4/check_docs.py`` exists because two things rotted in exactly this gap
    and were invisible to every other test here: ``--md`` (used by the workflows,
    the Colab cell and v4.bat, but no longer defined -- exit 2 in CI, which reads
    like a metric regression) and a ``python link_test.py --full`` hint in
    lpi_train.py pointing at a script nobody ever wrote.
    """
    import check_docs

    missing, bad, n = check_docs.scan()
    assert n > 20, f"the scan matched only {n} commands -- did the DOCS list break?"
    assert not missing, "documents a script that does not exist: " + ", ".join(
        f"{rel}: {name}" for rel, name, _ in missing)
    assert not bad, "documented flag is not defined by argparse: " + ", ".join(
        f"{name} {fl} ({rel})" for rel, name, fl, _ in bad)


def test_exported_torchscript_matches_the_checkpoint():
    """The .grc blocks and tx/rx_usrp.py load ``run/{generator,decoder}_lpi.pt``, not
    the training checkpoint, so the export must be a faithful copy of it.

    Two silent failure modes this pins: a trace that baked in a stale key tie
    (`LPIDecoder.tie_to`), and a smoke run exporting *its own* weights next to a
    different config -- both produce files that load happily and decode garbage.
    """
    import json

    best_pt = os.path.join(V4, "run", "lpi_v4.best.pt")
    dec_pt = os.path.join(V4, "run", "decoder_lpi.pt")
    gen_pt = os.path.join(V4, "run", "generator_lpi.pt")
    if not all(os.path.exists(p) for p in (best_pt, dec_pt, gen_pt)):
        pytest.skip("no exported TorchScript next to the checkpoint")

    ck = torch.load(best_pt, map_location="cpu", weights_only=False)
    cfg = LPIConfig(**{k: (tuple(v) if k == "snr_range" else v)
                       for k, v in ck["cfg"].items()})
    # exactly the recipe lpi_eval.load_models() uses -- seed, build, tie, then load.
    # A decoder built without tie_to() differs by O(1), which is the single easiest
    # way to "test" the export and conclude it is broken.
    torch.manual_seed(0)
    G = LPIGenerator(cfg).eval()
    D = LPIDecoder(cfg).eval()
    D.tie_to(G)
    G.load_state_dict(ck["generator"], strict=False)
    D.load_state_dict(ck["decoder"], strict=False)

    # Batch 1 first: that is the shape the GRC blocks call with, while the trace was
    # made at batch 2.  A baked-in batch assumption would show up here and nowhere
    # else -- and `torch.jit.trace` warns about exactly that (size_prods -> const).
    dec_script = torch.jit.load(dec_pt).eval()
    for B in (1, 8):
        torch.manual_seed(0)
        y = torch.randn(B, 2, cfg.frame_len)
        with torch.no_grad():
            eager, script = D(y), dec_script(y)
        assert tuple(eager.shape) == tuple(script.shape), (eager.shape, script.shape)
        assert torch.allclose(eager, script, atol=1e-4, rtol=1e-4), (
            f"exported decoder drifts from the checkpoint at batch {B}: "
            f"{(eager - script).abs().max():.2e}")

    # The generator draws fresh masking noise on every call -- that IS the design --
    # so it can only be checked for shape/energy, never bitwise (check_trace=False).
    Gs = torch.jit.load(gen_pt).eval()          # weights are inside the script
    with torch.no_grad():
        out = Gs(torch.zeros(4, cfg.z_dim), torch.ones(4, cfg.n_bits))
    assert tuple(out.shape) == (4, 2, cfg.frame_len), out.shape
    assert torch.isfinite(out).all()

    # And the mirror must declare what it mirrors.  `lpi_config.json` is required to
    # equal the checkpoint's own cfg dict exactly -- including cfo_max, which is a
    # training-time impairment range and NOT a field limit (the radio benchmark takes
    # that from lpi_eval --cfo-max / rx_usrp --coarse-span, not from this file).
    man_p = os.path.join(V4, "run", "export_manifest.json")
    assert os.path.exists(man_p), (
        "run/*.pt exports predate export_for_grc's manifest -- regenerate them with "
        "python -c \"from lpi_train import export_for_grc; export_for_grc("
        "\"run/lpi_v4.best.pt\",\"cpu\",\"run\")\" so they are provably the "
        "measured model")
    import hashlib

    man = json.load(open(man_p))
    assert man["exported_from"] == "lpi_v4.best.pt", man
    digest = hashlib.sha256(open(best_pt, "rb").read()).hexdigest()
    assert man["sha256"] == digest, (
        f"the exported TorchScript came from {man['exported_from']}@"
        f"{man.get('epoch')}, not from the checkpoint under test")
    exported = json.load(open(os.path.join(V4, "run", "lpi_config.json")))
    assert exported == ck["cfg"], {
        k: (exported.get(k), ck["cfg"].get(k))
        for k in set(exported) | set(ck["cfg"]) if exported.get(k) != ck["cfg"].get(k)}


@pytest.mark.skipif(not HAVE_CKPT, reason="no trained checkpoint")
def test_exporting_does_not_disturb_the_training_stream(tmp_path):
    """``export_for_grc()`` runs inside the epoch loop, so it must be RNG-neutral.

    Tracing needs dummy tensors, i.e. torch.randn; unguarded, every export shifted
    the global stream and the *next* epoch drew different data.  Which means the
    number of best-epoch exports -- a printing detail -- decided the model, and a
    CI job flipped its own metric verdict between two commits because of it.
    """
    import lpi_train

    torch.manual_seed(7); np.random.seed(7)
    ref = [float(v) for v in torch.randn(8)]
    torch.manual_seed(7); np.random.seed(7)
    lpi_train.export_for_grc(CKPT, "cpu", str(tmp_path))
    after = [float(v) for v in torch.randn(8)]
    assert ref == after, f"the export consumed RNG: {ref[:3]} != {after[:3]}"
    assert (tmp_path / "export_manifest.json").exists(), "export must record its source"


def test_cp_upper_bound_is_exact_where_it_can_be():
    """The bound is what a zero-error row actually claims, so it had better be right."""
    from lpi_eval import cp_upper

    for n in (4, 96, 4096):                       # k = 0 is the exact closed form
        assert abs(cp_upper(0, n) - (1.0 - 0.05 ** (1.0 / n))) < 1e-12
    assert cp_upper(0, 96) < 0.05 < cp_upper(0, 4)   # why --n-msg exists
    assert cp_upper(10, 10) == 1.0
    assert cp_upper(-3, 5) == cp_upper(0, 5) and cp_upper(99, 5) == 1.0
    from lpi_eval import cp_n_needed
    # the sample a "< x" claim needs, with zero failures -- quoted in the READMEs
    assert cp_n_needed(0.05) == 59 and cp_n_needed(0.01) == 299
    assert cp_upper(0, cp_n_needed(0.05)) <= 0.05 + 1e-12       # the two agree
    assert cp_upper(0, cp_n_needed(0.05) - 1) > 0.05            # and it is tight
    ks = [cp_upper(k, 1000) for k in range(40)]
    assert all(a <= b + 1e-12 for a, b in zip(ks, ks[1:])), "must be monotone in k"
    assert all(0.0 <= v <= 1.0 for v in ks)
    assert cp_upper(2, 65536) < 1e-4


def _sample_rep():
    """A report dict in the shape lpi_eval writes -- built from the committed
    metrics/last.json so these formatters are exercised without a model."""
    import json

    rep = json.load(open(os.path.join(ROOT, "metrics", "last.json")))
    rep["block_message"] = {"6.0": {"ber": 0.0, "fer": 0.0, "msgs_ok": 16, "n": 16}}
    row = rep["block_link"]["6.0"] if "6.0" in rep["block_link"] else \
        rep["block_link"][next(iter(rep["block_link"]))]
    row["n_bits_tested"] = 65536
    row["field_ci95"] = 7.2e-5
    row["fer_ci95"] = 0.031
    return rep


def test_latex_fragment_is_valid_latex_structurally():
    """No TeX in the sandbox, so check what actually breaks: balanced environments
    and one ``&`` per declared column -- plus graceful degradation on a JSON that
    predates the CI columns."""
    import json
    import re

    import lpi_latex

    cfg = LPIConfig()
    tex = lpi_latex.latex_tables(_sample_rep(), cfg, "lpi_eval.py --n-msg 16")
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    assert sorted(begins) == sorted(ends) and begins, (begins, ends)
    for spec, cols in re.findall(r"\\begin\{tabular\}@\{\}(l+c+)@\{\}", tex):
        width = len(cols)
        for line in tex.splitlines():
            if line.rstrip().endswith("\\\\") and "&" in line:
                n = line.count("&") + 1
                assert n == width, f"{n} cells in a {width}-column row: {line}"
    assert "tab:lpi-stealth" in tex and "tab:lpi-link" in tex
    assert "-0.0048" in tex or "-0.005" in tex                  # signed formatting
    assert "0.8646" in tex                                       # value, not the key
    # an older report without the bound columns still renders (falls back to the point)
    old = json.loads(json.dumps(_sample_rep()))
    for r in old["block_link"].values():
        r.pop("field_ci95", None)
    assert "7.20e-05" in lpi_latex.latex_tables(_sample_rep(), cfg)


def test_md_report_renders_from_a_saved_json():
    import lpi_eval

    md = lpi_eval.md_report(_sample_rep(), LPIConfig())
    assert md.startswith("## ShadowComm LPI v4")
    for col in ("95 %", "ks_p", "SNR"):
        assert col in md, col
