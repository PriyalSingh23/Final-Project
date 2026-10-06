"""Build a synthetic rx_raw.fc32 that mimics the over-the-air path:
real message bits -> exported TorchScript generator -> random sample offset,
CFO, gain, and AWGN. Then ota_verify.py must recover offset/CFO and BER."""
import numpy as np
import torch

FS = 32000
FRAMES = 400
OFFSET = 173                      # unknown to the verifier
CFO = 60.0                        # Hz, unknown to the verifier
PHI0 = np.deg2rad(117.0)          # constant LO phase, unknown to the verifier
SNR_DB = 15.0

msg_bits = np.unpackbits(np.fromfile('test_message.txt', dtype=np.uint8), bitorder='big')
need = FRAMES * 256
reps = int(np.ceil(need / len(msg_bits))) + 1
bits = np.tile(msg_bits, reps)[:need].reshape(FRAMES, 256)
b = torch.from_numpy(bits.astype(np.float32) * 2 - 1)          # +-1

G = torch.jit.load('generator_lpi.pt', map_location='cpu').eval()
frames = []
with torch.no_grad():
    for i in range(FRAMES):
        z = torch.randn(1, 64)
        x = G(z, b[i:i + 1]).numpy()[0]                        # (2,512)
        frames.append(x[0, :] + 1j * x[1, :])
frames = np.concatenate(frames)

# channel: gain + CFO + constant phase + carrier leakage + RX DC offset + AWGN
gain = 0.35
sig = frames * gain
n = np.arange(len(sig))
sig = sig * np.exp(1j * (2 * np.pi * CFO * n / FS + PHI0))
# TX carrier leakage: LO fraction leaking into the TX output -> line at CFO
leak = np.sqrt(2) * 0.02 * np.exp(1j * (2 * np.pi * CFO * n / FS + PHI0))
# RX DC offset (baseband constant, appears at 0 Hz at the RX)
dc = 0.01 + 0.01j
noise_std = gain * 10 ** (-SNR_DB / 20)
sig = sig + leak + dc
sig = sig + noise_std * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
sig = sig.astype(np.complex64)

cap = np.concatenate([np.zeros(OFFSET, dtype=np.complex64), sig])
cap.tofile('rx_raw.fc32')
print(f'synthetic capture: {len(cap)} samples, offset={OFFSET}, cfo={CFO} Hz, phi0={PHI0} rad, snr={SNR_DB} dB')
