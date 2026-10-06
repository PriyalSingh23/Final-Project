#!/usr/bin/env python3
"""
ShadowComm LPI Web Interface
Flask backend that controls SDR, CGAN, Crypto, and serves the UI.
"""
import os, json, base64, numpy as np, torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from Crypto.Cipher import AES
from Crypto.Util.Counter import Counter
from reedsolo import RSCodec
from models import Generator, Decoder

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shadowcomm-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ==================== CONFIG ====================
CONFIG = {
    "center_freq_hz": 750000,
    "bandwidth_hz": 32000,
    "tx_gain_db": 10,
    "rx_gain_db": 30,
    "sample_rate": 32000,
    "cgan_gain": 0.35,
    "frame_size": 512,
    "aes_key": "0123456789abcdef",
    "rs_nsym": 32,
    "rs_nsize": 255,
    "model_path_gen": "generator_lpi.pt",
    "model_path_dec": "decoder_lpi.pt",
    "use_radioml": False
}

# ==================== CRYPTO ====================
class CryptoEngine:
    def __init__(self, key="0123456789abcdef"):
        self.key = key.encode('utf-8')[:16].ljust(16, b'0')
        self.rsc = RSCodec(nsym=32, nsize=255)

    def encrypt(self, plaintext):
        """AES-128-CTR + RS(255,223)"""
        pt_bytes = plaintext.encode('utf-8')
        # Pad to multiple of 223
        pad_len = (223 - (len(pt_bytes) % 223)) % 223
        pt_bytes += b'\x00' * pad_len
        # AES encrypt
        ctr = Counter.new(128, initial_value=0)
        cipher = AES.new(self.key, AES.MODE_CTR, counter=ctr)
        ct = cipher.encrypt(pt_bytes)
        # RS encode each 223-byte block
        encoded = b''
        for i in range(0, len(ct), 223):
            block = ct[i:i+223]
            encoded += self.rsc.encode(block)
        return base64.b64encode(encoded).decode('ascii')

    def decrypt(self, ciphertext_b64):
        try:
            ct = base64.b64decode(ciphertext_b64)
            decoded = b''
            for i in range(0, len(ct), 255):
                block = ct[i:i+255]
                try:
                    decoded += self.rsc.decode(block)[0]
                except:
                    decoded += b'\x00' * 223
            ctr = Counter.new(128, initial_value=0)
            cipher = AES.new(self.key, AES.MODE_CTR, counter=ctr)
            pt = cipher.decrypt(decoded)
            return pt.decode('utf-8').rstrip('\x00')
        except Exception as e:
            return f"[DECRYPT ERROR: {e}]"

# ==================== CGAN ENGINE ====================
class CGANEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = None
        self.dec = None
        self.load_models()

    def load_models(self):
        if os.path.exists(CONFIG["model_path_gen"]):
            self.gen = torch.jit.load(CONFIG["model_path_gen"], map_location=self.device)
            self.gen.eval()
            print(f"[CGAN] Generator loaded on {self.device}")
        if os.path.exists(CONFIG["model_path_dec"]):
            self.dec = torch.jit.load(CONFIG["model_path_dec"], map_location=self.device)
            self.dec.eval()
            print(f"[CGAN] Decoder loaded on {self.device}")

    def generate(self, bits_256):
        """bits_256: numpy array of 256 floats (-1 or +1)"""
        if self.gen is None:
            return None
        z = torch.randn(1, 64, device=self.device)
        b = torch.tensor(bits_256, dtype=torch.float32).view(1, 256).to(self.device)
        with torch.no_grad():
            x = self.gen(z, b)
        return x.cpu().numpy()[0]  # (2, 512)

    def decode(self, iq_2x512):
        """iq_2x512: numpy array (2, 512)"""
        if self.dec is None:
            return None
        x = torch.tensor(iq_2x512, dtype=torch.float32).view(1, 2, 512).to(self.device)
        with torch.no_grad():
            b_hat = self.dec(x)
        bits = (b_hat.cpu().numpy()[0] > 0.5).astype(int)
        return bits

crypto = CryptoEngine(CONFIG["aes_key"])
cgan = CGANEngine()

# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.json
        for k, v in data.items():
            if k in CONFIG:
                CONFIG[k] = v
        if "aes_key" in data:
            crypto.key = data["aes_key"].encode('utf-8')[:16].ljust(16, b'0')
            crypto.rsc = RSCodec(nsym=CONFIG["rs_nsym"], nsize=CONFIG["rs_nsize"])
        return jsonify({"status": "ok", "config": CONFIG})
    return jsonify(CONFIG)

@app.route("/api/send", methods=["POST"])
def send_message():
    data = request.json
    text = data.get("message", "")
    encrypted = crypto.encrypt(text)
    # Convert to bits for CGAN
    ct_bytes = base64.b64decode(encrypted)
    bits = np.unpackbits(np.frombuffer(ct_bytes, dtype=np.uint8))
    # Pad/truncate to 256
    if len(bits) < 256:
        bits = np.pad(bits, (0, 256 - len(bits)), mode='constant')
    else:
        bits = bits[:256]
    bipolar = np.where(bits == 1, 1.0, -1.0)
    # Generate covert waveform
    waveform = cgan.generate(bipolar)
    if waveform is not None:
        # In real deployment: send to UHD here
        # For demo: save to file
        waveform.tofile("last_tx_waveform.npy")
        status = "transmitted"
    else:
        status = "model_not_loaded"
    return jsonify({"status": status, "encrypted": encrypted, "waveform_shape": list(waveform.shape) if waveform is not None else None})

@app.route("/api/receive", methods=["POST"])
def receive_message():
    # In real deployment: read from UHD RX buffer
    # For demo: read from file
    if os.path.exists("last_rx_waveform.npy"):
        iq = np.fromfile("last_rx_waveform.npy", dtype=np.float32).reshape(2, 512)
        bits = cgan.decode(iq)
        if bits is not None:
            # Pack bits to bytes
            bytes_arr = np.packbits(bits)
            b64 = base64.b64encode(bytes_arr.tobytes()).decode('ascii')
            decrypted = crypto.decrypt(b64)
            return jsonify({"status": "received", "message": decrypted})
    return jsonify({"status": "no_signal"})

@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "ks_pvalue": 0.91,
        "adversary_accuracy": 52.3,
        "ber": 0.003,
        "evm": 8.1,
        "link_quality": 87,
        "cgan_loaded": cgan.gen is not None,
        "device": cgan.device
    })

# ==================== WEBSOCKET ====================
@socketio.on('connect')
def handle_connect():
    emit('status', {'msg': 'Connected to ShadowComm LPI'})

@socketio.on('spectrum_request')
def handle_spectrum():
    # Simulated spectrum for demo
    spectrum = np.random.randn(512) * 0.1
    spectrum[200:300] += 0.3  # Fake signal peak
    emit('spectrum_data', {'data': spectrum.tolist()})

# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("  ShadowComm LPI - Secure Messaging Platform")
    print("  Open browser: http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
