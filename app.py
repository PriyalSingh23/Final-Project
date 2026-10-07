#!/usr/bin/env python3
"""ShadowComm LPI demonstration app.

The web app performs offline CGAN waveform generation and a simulated
model-to-model loopback. It does not control a USRP; LPI/BER claims must come
from the offline evaluation report and, ultimately, measured RF testing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
import secrets
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit
from Crypto.Cipher import AES
from Crypto.Util import Counter
from reedsolo import RSCodec

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_REPORT = Path(os.environ.get("CGAN_METRICS_REPORT", PROJECT_DIR / "lpi_metrics.json")).expanduser()
BITS_PER_FRAME = 256
SAMPLES_PER_FRAME = 512
MAX_MESSAGE_BYTES = 64 * 1024

app = Flask(__name__, template_folder=str(PROJECT_DIR))
app.config["SECRET_KEY"] = os.environ.get("SHADOWCOMM_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = MAX_MESSAGE_BYTES + 1024
socketio = SocketIO(app, cors_allowed_origins="*")

CONFIG = {
    "center_freq_hz": 750000,
    "bandwidth_hz": 32000,
    "tx_gain_db": 10,
    "rx_gain_db": 30,
    "sample_rate": 32000,
    "cgan_gain": 0.35,
    "frame_size": SAMPLES_PER_FRAME,
    "aes_key": os.environ.get("SHADOWCOMM_AES_KEY", "0123456789abcdef"),
    "rs_nsym": 64,
    "rs_nsize": 255,
    "model_path_gen": os.environ.get("CGAN_GENERATOR", "generator_lpi.pt"),
    "model_path_dec": os.environ.get("CGAN_DECODER", "decoder_lpi.pt"),
}


def resolve_model_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ==================== CRYPTO ====================
class CryptoEngine:
    def __init__(self, key: str = "0123456789abcdef"):
        self.set_key(key)

    def set_key(self, key: str) -> None:
        self.key = key.encode("utf-8")[:16].ljust(16, b"0")
        self.rsc = RSCodec(nsym=int(CONFIG["rs_nsym"]), nsize=int(CONFIG["rs_nsize"]))
        self.data_bytes = int(CONFIG["rs_nsize"]) - int(CONFIG["rs_nsym"])

    def encrypt(self, plaintext: str) -> str:
        """Encrypt UTF-8 text, apply the configured RS code, and return base64."""
        plain_bytes = plaintext.encode("utf-8")
        pad_len = (self.data_bytes - len(plain_bytes) % self.data_bytes) % self.data_bytes
        padded = plain_bytes + b"\x00" * pad_len
        counter = Counter.new(128, initial_value=0)
        cipher = AES.new(self.key, AES.MODE_CTR, counter=counter)
        encrypted = cipher.encrypt(padded)
        blocks = [
            self.rsc.encode(encrypted[start:start + self.data_bytes])
            for start in range(0, len(encrypted), self.data_bytes)
        ]
        # Interleave codewords by byte so a bad decoded IQ frame spreads its
        # symbol errors across several RS blocks instead of ruining one block.
        codewords = np.stack([np.frombuffer(block, dtype=np.uint8) for block in blocks])
        interleaved = codewords.T.reshape(-1).tobytes()
        return base64.b64encode(interleaved).decode("ascii")

    def decrypt(self, ciphertext_b64: str) -> str:
        try:
            coded = base64.b64decode(ciphertext_b64, validate=True)
            nsize = int(CONFIG["rs_nsize"])
            if not coded or len(coded) % nsize:
                raise ValueError("ciphertext is not a whole number of interleaved RS code blocks")
            nblocks = len(coded) // nsize
            interleaved = np.frombuffer(coded, dtype=np.uint8).reshape(nsize, nblocks)
            codewords = interleaved.T
            decoded = bytearray()
            for block in codewords:
                decoded.extend(self.rsc.decode(block.tobytes())[0])
            counter = Counter.new(128, initial_value=0)
            cipher = AES.new(self.key, AES.MODE_CTR, counter=counter)
            plaintext = cipher.decrypt(bytes(decoded))
            return plaintext.decode("utf-8").rstrip("\x00")
        except Exception as exc:
            return f"[DECRYPT ERROR: {exc}]"


# ==================== CGAN RUNTIME ====================
class CGANEngine:
    """Load TorchScript models and batch 256-bit message frames."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = None
        self.dec = None
        self.load_errors: dict[str, str] = {}
        self.manifest = None
        self.manifest_matches = False
        self.generated_frames = 0
        self._last_transmission = None
        self._lock = threading.RLock()
        self.load_models()

    def load_models(self) -> None:
        """Load/reload runtime artifacts; an absent model does not stop the app."""
        errors: dict[str, str] = {}
        loaded = {}
        for role, config_key in (("generator", "model_path_gen"), ("decoder", "model_path_dec")):
            path = resolve_model_path(CONFIG[config_key])
            if not path.is_file():
                errors[role] = f"model file not found: {path}"
                loaded[role] = None
                continue
            try:
                model = torch.jit.load(str(path), map_location=self.device).eval()
                loaded[role] = model
            except Exception as exc:  # keep the UI/API alive and expose the load error
                errors[role] = f"could not load {path}: {exc}"
                loaded[role] = None
        manifest = None
        manifest_matches = False
        manifest_path = PROJECT_DIR / "cgan_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                gen_path = resolve_model_path(CONFIG["model_path_gen"])
                dec_path = resolve_model_path(CONFIG["model_path_dec"])
                manifest_matches = bool(
                    loaded["generator"] is not None
                    and loaded["decoder"] is not None
                    and manifest.get("generator", {}).get("sha256") == file_sha256(gen_path)
                    and manifest.get("decoder", {}).get("sha256") == file_sha256(dec_path)
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors["manifest"] = f"could not validate model manifest: {exc}"
        with self._lock:
            self.gen = loaded["generator"]
            self.dec = loaded["decoder"]
            self.load_errors = errors
            self.manifest = manifest
            self.manifest_matches = manifest_matches

    def generate_frames(self, bipolar_frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Generate float32 IQ frames from an (N,256) array of -1/+1 symbols."""
        if self.gen is None:
            raise RuntimeError(self.load_errors.get("generator", "generator model is unavailable"))
        symbols = np.asarray(bipolar_frames, dtype=np.float32)
        if symbols.ndim == 1:
            symbols = symbols[None, :]
        if symbols.ndim != 2 or symbols.shape[1] != BITS_PER_FRAME:
            raise ValueError(f"expected (N,{BITS_PER_FRAME}) bipolar message frames")
        if not np.isfinite(symbols).all():
            raise ValueError("message bits contain NaN or infinity")
        if not np.isin(symbols, (-1.0, 1.0)).all():
            raise ValueError("message symbols must be encoded as -1 or +1")

        results = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(symbols), max(1, batch_size)):
                chunk = torch.from_numpy(np.ascontiguousarray(symbols[start:start + batch_size])).to(self.device)
                noise = torch.randn(len(chunk), 64, device=self.device)
                waveform = self.gen(noise, chunk)
                if waveform.shape != (len(chunk), 2, SAMPLES_PER_FRAME):
                    raise RuntimeError(f"generator returned unexpected shape {tuple(waveform.shape)}")
                values = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise RuntimeError("generator returned non-finite IQ values")
                results.append(values)
            self.generated_frames += len(symbols)
        return np.concatenate(results, axis=0)

    def decode_frames(self, iq_frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Decode (N,2,512) IQ frames into binary uint8 bit rows."""
        if self.dec is None:
            raise RuntimeError(self.load_errors.get("decoder", "decoder model is unavailable"))
        frames = np.asarray(iq_frames, dtype=np.float32)
        if frames.ndim == 2:
            frames = frames[None, ...]
        if frames.ndim != 3 or frames.shape[1:] != (2, SAMPLES_PER_FRAME):
            raise ValueError(f"expected (N,2,{SAMPLES_PER_FRAME}) IQ frames")
        if not np.isfinite(frames).all():
            raise ValueError("IQ frames contain NaN or infinity")

        decoded = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(frames), max(1, batch_size)):
                chunk = torch.from_numpy(np.ascontiguousarray(frames[start:start + batch_size])).to(self.device)
                probabilities = self.dec(chunk)
                if probabilities.shape != (len(chunk), BITS_PER_FRAME):
                    raise RuntimeError(f"decoder returned unexpected shape {tuple(probabilities.shape)}")
                decoded.append(probabilities.ge(0.5).to(torch.uint8).cpu().numpy())
        return np.concatenate(decoded, axis=0)

    def remember_transmission(self, iq_frames: np.ndarray, ciphertext_bytes: int) -> str:
        message_id = secrets.token_hex(8)
        with self._lock:
            self._last_transmission = {
                "message_id": message_id,
                "iq_frames": np.array(iq_frames, dtype=np.float32, copy=True),
                "ciphertext_bytes": int(ciphertext_bytes),
            }
        return message_id

    def last_transmission(self):
        with self._lock:
            if self._last_transmission is None:
                return None
            item = self._last_transmission
            return {
                **item,
                "iq_frames": item["iq_frames"].copy(),
            }

    def health(self) -> dict:
        return {
            "generator_loaded": self.gen is not None,
            "decoder_loaded": self.dec is not None,
            "device": str(self.device),
            "generated_frames": int(self.generated_frames),
            "manifest_matches_loaded_models": bool(self.manifest_matches),
            "checkpoint_epoch": self.manifest.get("checkpoint_epoch") if self.manifest else None,
            "load_errors": dict(self.load_errors),
            "transport": "offline_demo_only",
        }


crypto = CryptoEngine(CONFIG["aes_key"])
cgan = CGANEngine()


# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": "Expected a JSON object"}), 400

        reload_models = False
        for key, value in data.items():
            if key not in CONFIG:
                continue
            if key in ("model_path_gen", "model_path_dec"):
                if not isinstance(value, str) or not value.strip():
                    return jsonify({"status": "error", "error": f"{key} must be a non-empty path string"}), 400
                CONFIG[key] = value.strip()
                reload_models = True
            elif key == "aes_key":
                if not isinstance(value, str) or not value:
                    return jsonify({"status": "error", "error": "AES key must be a non-empty string"}), 400
                CONFIG[key] = value
                crypto.set_key(value)
            elif key in ("center_freq_hz", "bandwidth_hz", "tx_gain_db", "rx_gain_db", "sample_rate", "frame_size"):
                try:
                    CONFIG[key] = int(value)
                except (TypeError, ValueError):
                    return jsonify({"status": "error", "error": f"{key} must be an integer"}), 400
            elif key in ("rs_nsym", "rs_nsize"):
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    return jsonify({"status": "error", "error": f"{key} must be an integer"}), 400
                nsym = candidate if key == "rs_nsym" else int(CONFIG["rs_nsym"])
                nsize = candidate if key == "rs_nsize" else int(CONFIG["rs_nsize"])
                if not 1 <= nsym < nsize <= 255:
                    return jsonify({"status": "error", "error": "Require 1 <= rs_nsym < rs_nsize <= 255"}), 400
                CONFIG[key] = candidate
                crypto.set_key(CONFIG["aes_key"])
            elif key == "cgan_gain":
                try:
                    gain = float(value)
                except (TypeError, ValueError):
                    return jsonify({"status": "error", "error": "cgan_gain must be numeric"}), 400
                if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
                    return jsonify({"status": "error", "error": "cgan_gain must be between 0 and 1"}), 400
                CONFIG[key] = gain
        if reload_models:
            cgan.load_models()
        safe_config = {key: value for key, value in CONFIG.items() if key != "aes_key"}
        safe_config["aes_key_set"] = bool(CONFIG["aes_key"])
        return jsonify({"status": "ok", "config": safe_config, "models": cgan.health()})

    safe_config = {key: value for key, value in CONFIG.items() if key != "aes_key"}
    safe_config["aes_key_set"] = bool(CONFIG["aes_key"])
    return jsonify({"config": safe_config, "models": cgan.health()})


@app.route("/api/send", methods=["POST"])
def send_message():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("message"), str):
        return jsonify({"status": "error", "error": "message must be a string"}), 400
    text = data["message"]
    if not text:
        return jsonify({"status": "error", "error": "message cannot be empty"}), 400
    if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
        return jsonify({"status": "error", "error": f"message exceeds {MAX_MESSAGE_BYTES} bytes"}), 413
    if cgan.gen is None:
        return jsonify({"status": "model_not_loaded", "models": cgan.health()}), 503

    encrypted_b64 = crypto.encrypt(text)
    ciphertext = base64.b64decode(encrypted_b64)
    bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8), bitorder="big")
    frame_count = math.ceil(len(bits) / BITS_PER_FRAME)
    padded = np.pad(bits, (0, frame_count * BITS_PER_FRAME - len(bits)))
    binary_frames = padded.reshape(frame_count, BITS_PER_FRAME)
    bipolar_frames = binary_frames.astype(np.float32) * 2.0 - 1.0
    try:
        iq_frames = cgan.generate_frames(bipolar_frames)
    except Exception as exc:
        return jsonify({"status": "generation_error", "error": str(exc)}), 500

    message_id = cgan.remember_transmission(iq_frames, len(ciphertext))
    return jsonify({
        "status": "generated",
        "message_id": message_id,
        "transport": "simulated_only_not_transmitted",
        "encrypted": encrypted_b64,
        "frame_count": int(frame_count),
        "ciphertext_bytes": int(len(ciphertext)),
        "waveform_shape": list(iq_frames.shape),
        "waveform_url": "/api/last-waveform",
    })


@app.route("/api/last-waveform", methods=["GET"])
def last_waveform():
    transmission = cgan.last_transmission()
    if transmission is None:
        return jsonify({"status": "no_signal"}), 404
    buffer = BytesIO()
    np.save(buffer, transmission["iq_frames"], allow_pickle=False)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"cgan_iq_{transmission['message_id']}.npy",
    )


@app.route("/api/receive", methods=["POST"])
def receive_message():
    """Decode the last in-memory waveform as an explicitly simulated loopback."""
    transmission = cgan.last_transmission()
    if transmission is None:
        return jsonify({"status": "no_signal", "transport": "simulated_only"}), 404
    receive_data = request.get_json(silent=True)
    if receive_data is not None and not isinstance(receive_data, dict):
        return jsonify({"status": "error", "error": "Expected a JSON object"}), 400
    requested_id = (receive_data or {}).get("message_id")
    if requested_id and requested_id != transmission["message_id"]:
        return jsonify({"status": "not_found", "error": "message_id is not the last generated transmission"}), 404
    if cgan.dec is None:
        return jsonify({"status": "decoder_not_loaded", "models": cgan.health()}), 503

    try:
        decoded_frames = cgan.decode_frames(transmission["iq_frames"])
        recovered_bits = decoded_frames.reshape(-1)[:transmission["ciphertext_bytes"] * 8]
        recovered_bytes = np.packbits(recovered_bits, bitorder="big").tobytes()
        recovered_b64 = base64.b64encode(recovered_bytes).decode("ascii")
        message = crypto.decrypt(recovered_b64)
    except Exception as exc:
        return jsonify({"status": "decode_error", "error": str(exc)}), 500
    return jsonify({
        "status": "received",
        "transport": "simulated_model_loopback_not_radio",
        "message_id": transmission["message_id"],
        "frame_count": int(len(transmission["iq_frames"])),
        "message": message,
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Return measured offline metrics, or nulls instead of invented values."""
    result = {
        "ks_pvalue": None,
        "adversary_accuracy": None,
        "ber": None,
        "cyclo_ratio": None,
        "evm": None,
        "link_quality": None,
        "measurement_status": "not_measured",
        "measurement_scope": "Run test_metrics.py to create an offline synthetic-model report; this is not an OTA claim.",
        "report_generated_at": None,
        "report_checkpoint": None,
        **cgan.health(),
    }
    try:
        if METRICS_REPORT.is_file():
            report = json.loads(METRICS_REPORT.read_text(encoding="utf-8"))
            result["report_generated_at"] = report.get("generated_at")
            result["report_checkpoint"] = report.get("checkpoint")
            manifest_checkpoint_hash = (cgan.manifest or {}).get("checkpoint_sha256")
            report_matches = (
                cgan.manifest_matches
                and bool(report.get("checkpoint_sha256"))
                and report.get("checkpoint_sha256") == manifest_checkpoint_hash
            )
            if report_matches:
                values = report.get("metrics", {})
                result.update({
                    "ks_pvalue": values.get("ks_pvalue"),
                    "adversary_accuracy": values.get("adversary_accuracy_pct", values.get("adversary_accuracy")),
                    "ber": values.get("ber"),
                    "cyclo_ratio": values.get("cyclo_ratio"),
                    "measurement_status": "offline_report_loaded",
                    "measurement_scope": report.get("measurement_scope", "offline synthetic-model diagnostics; no RF/SDR channel"),
                    "report_passed": report.get("passed", {}).get("all"),
                })
            else:
                result["measurement_status"] = "stale_report"
                result["measurement_scope"] = "Metrics report does not match the currently exported generator/decoder; rerun test_metrics.py and export_for_grc.py."
    except (OSError, ValueError, TypeError) as exc:
        result["measurement_status"] = "report_error"
        result["report_error"] = str(exc)
    return jsonify(result)


# ==================== WEBSOCKET ====================
@socketio.on("connect")
def handle_connect():
    emit("status", {"msg": "Connected to the offline ShadowComm CGAN demo"})


@socketio.on("spectrum_request")
def handle_spectrum():
    # This endpoint is deliberately labelled as simulated: no SDR driver is wired in.
    spectrum = np.random.randn(512) * 0.1
    emit("spectrum_data", {"data": spectrum.tolist(), "simulated": True})


# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  ShadowComm LPI-CGAN - offline demo")
    print("  Transport: simulated; no SDR is connected by app.py")
    print(f"  Generator: {resolve_model_path(CONFIG['model_path_gen'])}")
    print(f"  Decoder:   {resolve_model_path(CONFIG['model_path_dec'])}")
    print("=" * 60)
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        allow_unsafe_werkzeug=True,  # local research demo; not a production server
    )
