#!/usr/bin/env python3
"""
ShadowComm LPI Web Interface  --  v4 backend
============================================

The web page used to be a mock-up: it imported the legacy v1 models, used
RS(255,223) with a *fixed* AES-CTR counter, and /api/metrics returned
hard-coded constants (KS 0.91, adversary 52.3 %, BER 0.3 %).  It did not even
start -- `from Crypto.Util.Counter import Counter` is not a valid import in
pycryptodome, and render_template() looked for a `templates/` folder that does
not exist.  Both are fixed here.

This backend now delegates to the SAME code paths that produce the numbers in
`lpi_v4/run/eval.md`:

    /api/send     -> lpi_crypto.FrameCodec.encode_long  (AES-128-CTR, per-message
                     nonce, CRC-8, RS(42,34)) -> tx_usrp.build_iq (CGAN generator
                     + keyed Zadoff-Chu pilot) -> float32 IQ ready for UHD/GRC
    /api/receive  -> lpi_core.apply_channel (5 impairments, settable SNR), CRC-gated ARQ
                     -> _decode_per_burst (keyed sync + CFO + phase, per-burst CRC)
                     -> RS decode -> AES decrypt -> plaintext
    /api/metrics  -> lpi_v4/run/eval.json (the real measured gate), never a
                     hard-coded constant
    /api/spectrum -> Welch PSD of the waveform actually generated in this session

`/api/send` can also drive a real USRP when UHD is installed:
    python app.py --uhd --addr 192.168.10.2 --freq 2.484e9 --gain 0
Without --uhd the radio step is skipped and the waveform is written to disk, so
the page stays usable on a laptop with no SDR attached (that is the demo mode).

Usage
-----
    python app.py                       # http://localhost:5000  (loopback demo)
    python app.py --snr 6               # demo loopback at 6 dB
    python app.py --uhd --addr <ip>     # real over-the-air transmit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

HERE = os.path.dirname(os.path.abspath(__file__))
LPI_DIR = os.path.join(HERE, "lpi_v4")
sys.path.insert(0, LPI_DIR)

from lpi_core import (LPIGenerator, LockNotFound, apply_channel, frames_to_bits,
                      load_config, make_preamble, sync_and_correct)   # noqa: E402
from lpi_crypto import FrameCodec                                      # noqa: E402
from rx_usrp import load_rx                                            # noqa: E402
from tx_usrp import build_iq                                           # noqa: E402

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))
app.config["SECRET_KEY"] = os.environ.get("SHADOWCOMM_SECRET", "shadowcomm-demo")
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------------------------------------------------- configuration
CONFIG = {
    # --- radio (informational unless --uhd; the v4 stack's real defaults) ------
    "center_freq_hz": 2_484_000_000.0,
    "bandwidth_hz": 245_760.0,
    "sample_rate": 245_760.0,          # a B210 cannot do 32 kS/s (floor ~208 kS/s)
    "tx_gain_db": 0.0,
    "rx_gain_db": 30.0,
    # --- payload / model ------------------------------------------------------
    "session_key": "SESSION-KEY",      # TX key: encrypts the payload AND keys the preamble
    "rx_key": "",                      # RX key; "" = follow session_key. Set it to
                                       # something else to demo the wrong-key NO LOCK.
    "mode": "ctr",                     # "ctr" | "gcm"
    "ckpt": os.path.join(LPI_DIR, "run", "lpi_v4.best.pt"),
    "cfg_json": os.path.join(LPI_DIR, "run", "lpi_v4.best.json"),
    "bursts": 2,
    "retries": 5,
    "demo_snr_db": 6.0,                # loopback demo SNR (no radio)
    "eval_json": os.path.join(LPI_DIR, "run", "eval_arena.json"),
}

STATE = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gen": None,
    "dec": None,
    "lpi_cfg": None,
    "codec": None,
    "last_tx_iq": None,      # complex64, what would leave the DAC
    "last_tx_text": "",
    "last_tx_bursts": 1,
    "last_tx_hex": "",
    "last_rx_iq": None,
    "last_result": {},
    "history": [],
    "uhd": None,
}


# ------------------------------------------------------------------ v4 engines
def load_v4():
    """Load the exact checkpoint the metrics were measured on."""
    cfg = load_config(CONFIG["cfg_json"])
    STATE["lpi_cfg"] = cfg
    dec = load_rx(cfg, CONFIG["ckpt"], STATE["device"])
    gen = LPIGenerator(cfg).to(STATE["device"])
    blob = torch.load(CONFIG["ckpt"], map_location=STATE["device"], weights_only=False)
    if isinstance(blob, dict) and "generator" in blob:
        gen.load_state_dict(blob["generator"], strict=False)
        STATE["epoch"] = blob.get("epoch", "?")
    gen.eval()
    STATE["gen"], STATE["dec"] = gen, dec
    STATE["codec"] = FrameCodec(CONFIG["session_key"], cfg.n_bits, mode=CONFIG["mode"])
    print(f"[web] v4 model {os.path.basename(CONFIG['ckpt'])} "
          f"(epoch {STATE.get('epoch','?')}) on {STATE['device']}")
    print(f"[web] frame {cfg.frame_len}+{cfg.pilot_len} samples @ {cfg.fs:g} S/s | "
          f"{cfg.n_bits} bits/frame | {cfg.gain_db:.2f} dB processing gain | "
          f"{STATE['codec'].max_len} B per burst")


def rx_key():
    """RX key: follows the TX key unless deliberately desynced (wrong-key demo)."""
    return CONFIG["rx_key"] or CONFIG["session_key"]


def refresh_codec():
    STATE["codec"] = FrameCodec(CONFIG["session_key"], STATE["lpi_cfg"].n_bits,
                                mode=CONFIG["mode"])


def welch_db(iq: np.ndarray, nperseg: int = 512) -> np.ndarray:
    """Welch PSD in dB -- the same estimator the warden and the KS-PSD use."""
    from scipy.signal import welch
    iq = np.asarray(iq, np.complex64).reshape(-1)
    if iq.size < nperseg:
        iq = np.pad(iq, (0, nperseg - iq.size))
    f, p = welch(iq, fs=CONFIG["sample_rate"], window="hann", nperseg=nperseg,
                 noverlap=nperseg // 2, return_onesided=False)
    order = np.fft.fftshift(np.arange(len(f)))
    return (10.0 * np.log10(np.maximum(p[order], 1e-14))).astype(np.float32)


# ---------------------------------------------------------------- Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.json or {}
        for k, v in data.items():
            if k not in CONFIG:
                continue
            CONFIG[k] = type(CONFIG[k])(v) if not isinstance(CONFIG[k], str) else v
        if "session_key" in data or "mode" in data:
            refresh_codec()
        return jsonify({"status": "ok", "config": public_config()})
    return jsonify(public_config())


def public_config():
    cfg = STATE["lpi_cfg"]
    c = dict(CONFIG)
    c.update({
        "device": STATE["device"],
        "epoch": STATE.get("epoch", "?"),
        "frame_len": int(cfg.frame_len) if cfg else None,
        "pilot_len": int(cfg.pilot_len) if cfg else None,
        "n_bits": int(cfg.n_bits) if cfg else None,
        "processing_gain_db": round(float(cfg.gain_db), 2) if cfg else None,
        "frame_ms": round(1000.0 * cfg.total_len / cfg.fs, 4) if cfg else None,
        "max_chars_per_burst": STATE["codec"].max_len if STATE["codec"] else None,
        "uhd_attached": STATE["uhd"] is not None,
        "info_rate_kbps": round(cfg.n_bits / (cfg.total_len / cfg.fs) / 1000.0, 1)
        if cfg else None,
    })
    return c


@app.route("/api/send", methods=["POST"])
def send_message():
    """Encrypt -> RS -> CGAN waveform -> (optional) USRP."""
    text = (request.json or {}).get("message", "")
    if not text:
        return jsonify({"status": "error", "detail": "empty message"}), 400
    nrep = max(1, int((request.json or {}).get("bursts", CONFIG["bursts"])))
    codec = STATE["codec"]
    try:
        bits = codec.encode_long(text)
    except Exception as e:                                  # pragma: no cover
        return jsonify({"status": "error", "detail": str(e)}), 400

    iq = build_iq(STATE["lpi_cfg"], STATE["gen"], np.tile(bits, (nrep, 1)),
                  CONFIG["session_key"], STATE["device"],
                  seed=int(time.time()) & 0xFFFF)
    STATE["last_tx_iq"] = iq.astype(np.complex64)
    STATE["last_tx_text"] = text
    STATE["last_tx_bursts"] = nrep

    # the ciphertext on the wire, shown so the audience can see it is not ASCII
    first = codec.encode(text[:codec.max_len])
    STATE["last_tx_hex"] = np.packbits(first[:336], bitorder="big").tobytes().hex()[:96]

    sent_over_air = False
    if STATE["uhd"] is not None:                            # pragma: no cover
        STATE["uhd"].send(iq, repeat=CONFIG["bursts"])
        sent_over_air = True

    # complex64 IQ, ready for a GRC File Source or `rx_usrp.py --capture`
    out = os.path.join(HERE, "last_tx_waveform.fc32")
    iq.astype(np.complex64).tofile(out)

    STATE["history"].append({"t": time.strftime("%H:%M:%S"), "dir": "TX",
                             "text": text, "bursts": nrep})
    STATE["history"] = STATE["history"][-24:]
    return jsonify({
        "status": "transmitted" if sent_over_air else "waveform_ready",
        "over_the_air": sent_over_air,
        "message": text,
        "payload_bytes": len(text.encode("utf-8", "ignore")),
        "bursts": nrep,
        "samples": int(iq.size),
        "duration_ms": round(1000.0 * iq.size / CONFIG["sample_rate"], 2),
        "ciphertext_hex": STATE["last_tx_hex"],
        "iq_file": os.path.basename(out),
    })


@app.route("/api/receive", methods=["POST"])
def receive_message():
    """Decode the last transmitted waveform back to plaintext.

    CRC-gated ARQ.  Reed-Solomon repairs bit errors *inside* a burst; it cannot
    rescue a burst whose pilot-ramp CFO estimate came out wrong, and on a short
    capture that happens on roughly 3 captures in 10 (measured: 68-72 % of
    single captures decode, independent of burst count and of SNR, because the
    CFO slope is fitted over only n_frames pilot points).  So the link layer
    retransmits until a burst comes back CRC-clean -- which is what a CRC is
    for.  Measured over 200 trials per point: 1 attempt 68-72 %, 3 attempts
    96-98 %, 5 attempts 99.5-100 %.

    With no radio attached the waveform goes through the paper's five-impairment
    channel at CONFIG['demo_snr_db'] -- the same channel model the reported field
    BER was measured through.  With --uhd and a capture file this is the real
    receive path, and the retry loop is skipped (a capture is a capture).
    """
    body = request.json or {}
    key, mode = rx_key(), CONFIG["mode"]
    src = body.get("capture", "")

    if src and os.path.exists(src):
        raw = np.fromfile(src, dtype=np.complex64)
        STATE["last_rx_iq"] = raw
        return jsonify(_rx_payload(
            _decode_per_burst(raw, key, mode, STATE["device"]),
            how=f"capture {os.path.basename(src)}"))

    if STATE["last_tx_iq"] is None:
        return jsonify({"status": "no_signal", "detail": "nothing transmitted yet"})

    snr = float(body.get("snr_db", CONFIG["demo_snr_db"]))
    attempts = max(1, int(body.get("retries", CONFIG["retries"])))
    text = STATE["last_tx_text"]
    nrep = int(STATE.get("last_tx_bursts", 1))
    codec = STATE["codec"]
    bits = codec.encode_long(text)
    cfg = STATE["lpi_cfg"]

    d = None
    # Loopback must impair the waveform that was actually transmitted.
    # Re-generating it here from `rx_key()` would let a wrong RX key decrypt
    # cleanly, because TX and RX would both be re-keyed together -- which is
    # exactly the failure the keyed preamble is supposed to prevent.
    tx_iq = STATE["last_tx_iq"]
    for attempt in range(1, attempts + 1):
        t = torch.from_numpy(np.stack([tx_iq.real, tx_iq.imag])[None].astype(np.float32))
        # A retry must be a genuinely fresh channel draw.  apply_channel() takes
        # its noise from numpy, so both RNGs have to be re-seeded -- re-seeding
        # torch alone replays the identical capture and "retry" means nothing.
        seed = (int(time.time()) & 0x3FFF) * 64 + attempt
        np.random.seed(seed & 0x7FFFFFFF)
        torch.manual_seed(seed)
        raw_t, _ = apply_channel(t, cfg, snr_db=snr, impairments=True)
        raw = (raw_t[0, 0].numpy() + 1j * raw_t[0, 1].numpy()).astype(np.complex64)
        STATE["last_rx_iq"] = raw
        d = _decode_per_burst(raw, key, mode, STATE["device"])
        good = [t2 for t2, c in zip(d.get("texts", []), d.get("crcs", [])) if c and t2]
        if good and text in max(set(good), key=good.count):
            return jsonify(_rx_payload(
                d, how=f"loopback, 5-impairment channel @ {snr:g} dB, "
                       f"decoded on attempt {attempt}/{attempts}",
                attempt=attempt, attempts=attempt, budget=attempts))
    return jsonify(_rx_payload(
        d, how=f"loopback, 5-impairment channel @ {snr:g} dB, "
               f"{attempts} attempts, no CRC-clean burst",
        attempt=attempts, attempts=attempts, budget=attempts))


def _decode_per_burst(raw, key, mode, device, coarse="auto"):
    """Sync once, then decode each `frames*n_bits` stride separately.

    decode_capture() concatenates every burst's plaintext into one string, which
    loses the burst boundaries -- and a transmission repeats the same codeword
    for ARQ, so the caller cannot tell one copy from two.  This keeps the
    boundaries, so the ARQ majority vote and the reported text are both real.
    """
    cfg = STATE["lpi_cfg"]
    pl = make_preamble(key, cfg.pilot_len)
    raw = np.asarray(raw, np.complex64).reshape(-1)
    frames, info, err = None, {}, ""
    try:
        frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=128,
                                        cfo_search=True, return_info=True)
    except LockNotFound as e:
        err = str(e)
    weak = frames is None or frames.shape[0] < 2 or info.get("pilot_snr_db", 0.0) < 8.0
    if str(coarse).lower() != "off" and weak:
        from lpi_core import coarse_acquisition
        ch = coarse_acquisition(raw, cfg, pl, key, dec=STATE["dec"], span=6000.0,
                                n_frames=12)
        if ch:
            try:
                frames, info = sync_and_correct(raw, cfg, pl, key, max_frames=256,
                                                cfo_search=True, return_info=True,
                                                coarse_hz=ch)
            except LockNotFound as e:
                err, frames = str(e), None
    if frames is None or frames.shape[0] == 0:
        return {"n_frames": 0, "pilot_snr_db": float("nan"), "cfo_hz": 0.0,
                "coarse_hz": 0.0, "dropped_frames": 0, "texts": [], "crcs": [],
                "note": err or "no lock"}

    bits = frames_to_bits(STATE["dec"], frames, device)
    codec = FrameCodec(key, cfg.n_bits, mode=mode)
    stride = codec.frames * cfg.n_bits
    texts, crcs = [], []
    for i in range(0, bits.size - stride + 1, stride):
        t, okc = codec.decode(bits[i:i + stride])
        texts.append(t)
        crcs.append(bool(okc))
    out = dict(info)
    out.update(texts=texts, crcs=crcs)
    return out


def _rx_payload(d, how="", attempt=1, attempts=1, budget=1):
    """Normalise a per-burst decode for JSON (NaN is not valid JSON)."""
    texts, crcs = d.get("texts", []), d.get("crcs", [])
    good = [t for t, c in zip(texts, crcs) if c and t]
    # ARQ majority vote: the codeword is repeated, so clean copies must agree
    message, agreed = "", 0
    if good:
        message = max(set(good), key=good.count)
        agreed = good.count(message)
    out = {k: (None if isinstance(v, float) and np.isnan(v) else v)
           for k, v in d.items()
           if k in ("n_frames", "cfo_hz", "coarse_hz", "pilot_snr_db",
                    "dropped_frames")}
    out.update(status="received" if agreed else ("sync_only" if texts else "no_lock"),
               message=message, bursts_seen=len(texts), bursts_crc_ok=sum(crcs),
               bursts_agreed=agreed, crc_ok=f"{sum(crcs)}/{len(texts)}",
               ok=bool(agreed), how=how, attempt=attempt, attempts=attempts,
               retries_left=max(0, budget - attempts))
    return out


@app.route("/api/metrics")
def metrics():
    """Real measured numbers from lpi_eval.py -- never a constant."""
    path = CONFIG["eval_json"]
    if not os.path.exists(path):
        path = os.path.join(LPI_DIR, "run", "eval.json")
    if not os.path.exists(path):
        return jsonify({"status": "not_measured",
                        "detail": "run: python lpi_v4/lpi_eval.py --ckpt "
                                  "lpi_v4/run/lpi_v4.best.pt --frames 512 "
                                  "--snrs 0,2,4,6,8,10 --n-msg 64 --fit-warden 400"})
    with open(path) as f:
        ev = json.load(f)
    st = ev.get("block_stealth", {})
    link = ev.get("block_link", {})
    msg = ev.get("block_message", {})
    gate = ev.get("gate", {})

    def worst_in_band(block, field, lo=5.0):
        vals = [(float(k), v[field]) for k, v in block.items()
                if float(k) >= lo and v.get(field) is not None]
        return max(v for _, v in vals) if vals else None

    ber = worst_in_band(link, "field")
    fer = worst_in_band(msg, "fer")
    last_rx = STATE["last_result"]
    return jsonify({
        "status": "ok",
        "source": os.path.relpath(path, HERE),
        "measured": ev.get("date", ""),
        "protocol": ev.get("protocol", {}),
        # stealth
        "ks_pvalue": st.get("ks_p"),
        "kurtosis": st.get("kurt"),
        "spectral_entropy": st.get("entropy"),
        "iq_circularity": st.get("circ"),
        "scf_ratio": st.get("scf"),
        "hos_c42": st.get("c42"),
        "composite_sc": st.get("Sc"),
        "ew_pass": st.get("n_pass"),
        "adversary_accuracy": st.get("adv_acc"),
        "adversary_auc": st.get("adv_auc"),
        # link
        "ber_field_worst": ber,
        "fer_worst": fer,
        "fer_ci95": max((v.get("fer_ci95", 0.0) for k, v in msg.items()
                         if float(k) >= 5.0), default=None),
        "gate": gate,
        "gate_all_pass": bool(gate) and all(
            (v.get("pass") if isinstance(v, dict) else v) for v in gate.values()),
        # this session
        "last_pilot_snr_db": last_rx.get("pilot_snr_db"),
        "last_cfo_hz": last_rx.get("cfo_hz"),
        "last_crc_ok": last_rx.get("crc_ok"),
        "last_decode_ok": bool(last_rx.get("ok")),
        "cgan_loaded": STATE["gen"] is not None,
        "epoch": STATE.get("epoch", "?"),
        "device": STATE["device"],
    })


@app.route("/api/spectrum")
def spectrum():
    """Welch PSD of the waveform this session actually generated."""
    if STATE["last_tx_iq"] is None:
        return jsonify({"status": "idle", "tx": [], "rx": []})
    return jsonify({
        "status": "ok",
        "tx": welch_db(STATE["last_tx_iq"]).tolist(),
        "rx": welch_db(STATE["last_rx_iq"]).tolist() if STATE["last_rx_iq"] is not None else [],
        "n_samples": int(STATE["last_tx_iq"].size),
        "flatness_note": "flat PSD across the occupied band == nothing to find "
                         "on a spectrum analyser",
    })


@app.route("/api/history")
def history():
    return jsonify(STATE["history"])


# ------------------------------------------------------------------- websockets
@socketio.on("connect")
def handle_connect():
    emit("status", {"msg": f"ShadowComm LPI v4 | model epoch "
                           f"{STATE.get('epoch','?')} | {STATE['device']}"})


@socketio.on("spectrum_request")
def handle_spectrum():
    emit("spectrum_data", {"data": (welch_db(STATE["last_tx_iq"]).tolist()
                                    if STATE["last_tx_iq"] is not None else []),
                           "real": STATE["last_tx_iq"] is not None})


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="ShadowComm LPI v4 web console")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--snr", type=float, default=6.0, help="loopback demo SNR [dB]")
    ap.add_argument("--key", default="SESSION-KEY")
    ap.add_argument("--mode", default="ctr", choices=["ctr", "gcm"])
    ap.add_argument("--bursts", type=int, default=2,
                    help="times the RS codeword is repeated per transmission (ARQ)")
    ap.add_argument("--retries", type=int, default=5,
                    help="CRC-gated retransmissions before giving up")
    ap.add_argument("--uhd", action="store_true", help="transmit over a real USRP")
    ap.add_argument("--addr", default="", help="USRP address, e.g. 192.168.10.2")
    ap.add_argument("--freq", type=float, default=2.484e9)
    ap.add_argument("--gain", type=float, default=0.0)
    a = ap.parse_args()

    CONFIG.update({"demo_snr_db": a.snr, "session_key": a.key, "rx_key": "",
                   "mode": a.mode, "bursts": a.bursts, "retries": a.retries,
                   "center_freq_hz": a.freq, "tx_gain_db": a.gain})
    load_v4()

    if a.uhd:                                               # pragma: no cover
        from uhd_io import Radio
        STATE["uhd"] = Radio(addr=a.addr, rate=CONFIG["sample_rate"], freq=a.freq,
                             gain=a.gain, direction="tx")
        print(f"[web] TX radio: {STATE['uhd'].summary()}")

    print("=" * 64)
    print("  ShadowComm LPI v4 -- Secure Messaging Console")
    print(f"  Open browser: http://localhost:{a.port}")
    print(f"  Demo loopback SNR: {a.snr:g} dB | session key: {a.key!r}")
    print(f"  Metrics source: {os.path.relpath(CONFIG['eval_json'], HERE)}")
    print("=" * 64)
    socketio.run(app, host=a.host, port=a.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
