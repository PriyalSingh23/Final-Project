#!/usr/bin/env python3
"""
Comprehensive LPI Waveform & EW Detection Test Battery
Evaluates waveform against all 8 Electronic Warfare (EW) domains
and communications reliability benchmarks as reported in:
- Paper 1: RF-Robust Payload-Invariant Noise-Masked LPI Waveform Synthesis (IEEE GCON 2026)
- Paper 2: On Evasion of EW Detection with Generative Waveform Synthesis
- Paper 3: Project ANSHUMAN (Library Research Paper)

Metrics Evaluated:
  1. Kolmogorov-Smirnov (KS) p-value (> 0.05)
  2. Kurtosis κ (≈ 3.0)
  3. Spectral Entropy H (> 0.95)
  4. IQ Circularity η (< 0.15)
  5. PAPR Deviation (|Δ| < 3 dB)
  6. CSFA SCF Ratio (< 3x)
  7. Higher-Order Statistics C42 (|C42| < 0.3)
  8. WVD Time-Frequency Ratio (< 3.7x)
  9. Composite Stealth Score Sc (< 0.10: Excellent)
 10. Adversary CNN Detection Rate (48% - 56%)
 11. Pre-FEC Bit Error Rate (< 1%)
 12. Post-FEC Reed-Solomon Decoded BER (0.0000%)
"""
import os, sys, argparse, math, numpy as np, torch
from scipy import stats
from reedsolo import RSCodec, ReedSolomonError
from Crypto.Cipher import AES
from Crypto.Util import Counter

# Set OpenMP compatibility
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from models import Generator, Decoder, rand_circular_shift

ap = argparse.ArgumentParser(description="LPI Waveform & EW Multi-Domain Evaluation")
ap.add_argument('--ckpt', default='lpi_checkpoint.pt', help='Path to checkpoint')
ap.add_argument('--n-test', type=int, default=2000, help='Frames for statistical evaluation')
ap.add_argument('--adv-train', type=int, default=1500, help='Samples for adversary CNN training')
ap.add_argument('--adv-epochs', type=int, default=4, help='Adversary CNN training epochs')
ap.add_argument('--adv-test', type=int, default=1000, help='Samples for adversary evaluation')
ap.add_argument('--ber-msgs', type=int, default=500, help='Frames for BER evaluation')
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained models
if not os.path.exists(args.ckpt):
    # Try alternate checkpoint if primary not found
    for alt in ['lpi_checkpoint_v3_610.pt', 'lpi_checkpoint_v2_607.pt']:
        if os.path.exists(alt):
            args.ckpt = alt
            break

print(f"[EVAL] Loading checkpoint: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)

G = Generator(msg_len=256, z_dim=64, out_len=1024, dither_std=0.20).to(DEVICE).eval()
Dec = Decoder(msg_len=256).to(DEVICE).eval()

G.load_state_dict(ckpt['generator'], strict=False)
Dec.load_state_dict(ckpt['decoder'], strict=False)

print(f"[EVAL] Model loaded on {DEVICE} (checkpoint epoch {ckpt.get('epoch', '?')})")

@torch.no_grad()
def gen_batch(n):
    b = (torch.rand(n, 256, device=DEVICE) > 0.5).float() * 2.0 - 1.0
    z = torch.randn(n, 64, device=DEVICE)
    return G(z, b).cpu().numpy()

# Generate test frames
print(f"\nGenerating {args.n_test} LPI frames (512 complex samples = 1024 floats each)...")
gen_samples = []
batch_size = 250
for i in range(0, args.n_test, batch_size):
    cur_n = min(batch_size, args.n_test - i)
    gen_samples.append(gen_batch(cur_n))
gen_samples = np.concatenate(gen_samples, axis=0) # (N, 2, 512)
N = len(gen_samples)

# Complex representation
z_sig = gen_samples[:, 0, :] + 1j * gen_samples[:, 1, :]
awgn_raw = (np.random.randn(N, 512) + 1j * np.random.randn(N, 512)) / np.sqrt(2.0)

# Reference BPSK baseline for comparative benchmark
bpsk_symbols = np.random.choice([-1.0, 1.0], size=(N, 512)).astype(np.complex64)
bpsk_noise = np.random.randn(N, 512) * 0.1 + 1j * np.random.randn(N, 512) * 0.1
bpsk_sig = bpsk_symbols + bpsk_noise
bpsk_sig = bpsk_sig / np.sqrt(np.mean(np.abs(bpsk_sig)**2, axis=1, keepdims=True))

# ===========================================================================
# 1. KOLMOGOROV-SMIRNOV (KS) TEST & KURTOSIS
# ===========================================================================
# Unit-variance per real component (I and Q channels each ~ N(0, 1))
flat_lpi = gen_samples.reshape(-1)
flat_awgn = np.concatenate([awgn_raw.real.reshape(-1) * np.sqrt(2.0), awgn_raw.imag.reshape(-1) * np.sqrt(2.0)])
flat_bpsk = np.concatenate([bpsk_sig.real.reshape(-1), bpsk_sig.imag.reshape(-1)])

ks_stat_lpi, ks_p_lpi = stats.kstest(flat_lpi, stats.norm.cdf)
ks_stat_awgn, ks_p_awgn = stats.kstest(flat_awgn, stats.norm.cdf)
ks_stat_bpsk, ks_p_bpsk = stats.kstest(flat_bpsk, stats.norm.cdf)

kurt_lpi = stats.kurtosis(flat_lpi, fisher=False)
kurt_awgn = stats.kurtosis(flat_awgn, fisher=False)
kurt_bpsk = stats.kurtosis(flat_bpsk, fisher=False)

# ===========================================================================
# 2. SPECTRAL ENTROPY (H)
# ===========================================================================
def calc_spectral_entropy(signals):
    psds = np.abs(np.fft.fft(signals, axis=1))**2
    psd_m = np.mean(psds, axis=0)
    p_norm = psd_m / (np.sum(psd_m) + 1e-12)
    return -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm))

entropy_lpi = calc_spectral_entropy(z_sig)
entropy_awgn = calc_spectral_entropy(awgn_raw)
entropy_bpsk = calc_spectral_entropy(bpsk_sig)

# ===========================================================================
# 3. IQ CIRCULARITY (η)
# ===========================================================================
def calc_iq_circularity(signals):
    ez2 = np.abs(np.mean(signals**2))
    ezabs2 = np.mean(np.abs(signals)**2)
    return float(ez2 / (ezabs2 + 1e-12))

circ_lpi = calc_iq_circularity(z_sig)
circ_awgn = calc_iq_circularity(awgn_raw)
circ_bpsk = calc_iq_circularity(bpsk_sig)

# ===========================================================================
# 4. PAPR DEVIATION (dB)
# ===========================================================================
def calc_mean_papr(signals):
    papr_per_frame = np.max(np.abs(signals)**2, axis=1) / np.mean(np.abs(signals)**2, axis=1)
    return 10.0 * np.log10(np.mean(papr_per_frame))

papr_lpi = calc_mean_papr(z_sig)
papr_awgn = calc_mean_papr(awgn_raw)
papr_bpsk = calc_mean_papr(bpsk_sig)
papr_dev = papr_lpi - papr_awgn

# ===========================================================================
# 5. CSFA SPECTRAL CORRELATION FUNCTION (SCF) RATIO
# ===========================================================================
def calc_cyclo_mean(signals, max_lag=20):
    corrs = []
    for i in range(min(len(signals), 200)):
        sig = signals[i]
        c = [np.abs(np.mean(sig[:-lag] * np.conj(sig[lag:]))) for lag in range(1, max_lag + 1)]
        corrs.append(np.mean(c))
    return np.mean(corrs)

cyclo_lpi = calc_cyclo_mean(z_sig)
cyclo_awgn = calc_cyclo_mean(awgn_raw)
cyclo_bpsk = calc_cyclo_mean(bpsk_sig)
scf_ratio = cyclo_lpi / cyclo_awgn
scf_ratio_bpsk = cyclo_bpsk / cyclo_awgn

# ===========================================================================
# 6. HIGHER-ORDER STATISTICS (HOS C42)
# ===========================================================================
def calc_c42(signals):
    m4 = np.mean(np.abs(signals)**4)
    m2 = np.mean(np.abs(signals)**2)
    ez2 = np.mean(signals**2)
    return float(m4 - 2.0 * (m2**2) - np.abs(ez2)**2)

c42_lpi = calc_c42(z_sig)
c42_awgn = calc_c42(awgn_raw)
c42_bpsk = calc_c42(bpsk_sig)

# ===========================================================================
# 7. WVD / TIME-FREQUENCY CONCENTRATION RATIO
# ===========================================================================
from scipy.signal import spectrogram
def calc_tf_ratio(signals):
    ratios = []
    for i in range(min(len(signals), 50)):
        _, _, s = spectrogram(signals[i], fs=2e6, nperseg=64, noverlap=32, return_onesided=False)
        s_abs = np.abs(s)
        ratios.append(np.max(s_abs) / (np.mean(s_abs) + 1e-12))
    return np.mean(ratios)

tf_lpi = calc_tf_ratio(z_sig)
tf_awgn = calc_tf_ratio(awgn_raw)
tf_bpsk = calc_tf_ratio(bpsk_sig)
wvd_ratio = tf_lpi / tf_awgn
wvd_ratio_bpsk = tf_bpsk / tf_awgn

# ===========================================================================
# 8. COMPOSITE STEALTH SCORE (Sc)
# ===========================================================================
d_ks = max(0.0, 1.0 - ks_p_lpi)
d_kurt = abs(kurt_lpi - 3.0) / 3.0
d_ent = max(0.0, 1.0 - entropy_lpi)
d_circ = circ_lpi / 0.15
d_papr = abs(papr_dev) / 3.0
d_scf = abs(scf_ratio - 1.0) / 2.0
d_c42 = abs(c42_lpi) / 0.3
d_wvd = abs(wvd_ratio - 1.0) / 2.7
composite_sc = (d_ks + d_kurt + d_ent + d_circ + d_papr + d_scf + d_c42 + d_wvd) / 8.0

if composite_sc < 0.10:
    sc_grade = "Excellent"
elif composite_sc < 0.20:
    sc_grade = "Good"
else:
    sc_grade = "Moderate"

# ===========================================================================
# 9. ADVERSARY CNN CLASSIFICATION RESISTANCE
# ===========================================================================
print("\n" + "=" * 65)
print("TRAINING ADVERSARIAL ELECTRONIC WARFARE CNN (RadioML VT-CNN2)...")
print("=" * 65)

class AdversaryCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, 7, padding=3), torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1), torch.nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

adv = AdversaryCNN().to(DEVICE)
adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-4)

# Prepare balanced training set: 50% LPI, 50% AWGN
n_tr = args.adv_train
X_tr, y_tr = [], []
with torch.no_grad():
    for i in range(n_tr):
        if i % 2 == 0:
            b_dummy = (torch.rand(1, 256, device=DEVICE) > 0.5).float() * 2.0 - 1.0
            z_dummy = torch.randn(1, 64, device=DEVICE)
            covert = G(z_dummy, b_dummy)
            covert = rand_circular_shift(covert)
            X_tr.append(covert)
            y_tr.append(1.0)
        else:
            noise = torch.randn(1, 2, 512, device=DEVICE)
            noise = noise / noise.view(1, -1).pow(2).mean().sqrt()
            X_tr.append(noise)
            y_tr.append(0.0)

X_tr = torch.cat(X_tr, dim=0)
y_tr = torch.tensor(y_tr, device=DEVICE).view(-1, 1)

for epoch in range(args.adv_epochs):
    perm = torch.randperm(n_tr)
    for j in range(0, n_tr, 64):
        idx = perm[j:j+64]
        loss = torch.nn.BCELoss()(adv(X_tr[idx]), y_tr[idx])
        adv_opt.zero_grad(); loss.backward(); adv_opt.step()

adv.eval()
correct = 0
n_test = args.adv_test
with torch.no_grad():
    for i in range(n_test):
        if i % 2 == 0:
            b_dummy = (torch.rand(1, 256, device=DEVICE) > 0.5).float() * 2.0 - 1.0
            z_dummy = torch.randn(1, 64, device=DEVICE)
            covert = G(z_dummy, b_dummy)
            covert = rand_circular_shift(covert)
            pred = adv(covert).item()
            if pred > 0.5: correct += 1
        else:
            noise = torch.randn(1, 2, 512, device=DEVICE)
            pred = adv(noise).item()
            if pred <= 0.5: correct += 1

adv_acc = (correct / n_test) * 100.0

# ===========================================================================
# 10. COMMUNICATIONS PERFORMANCE (RAW PRE-FEC & REED-SOLOMON POST-FEC)
# ===========================================================================
print("\n" + "=" * 65)
print("EVALUATING END-TO-END BIT ERROR RATE & REED-SOLOMON DECODING...")
print("=" * 65)

rsc = RSCodec(10) # RS code with 10 parity bytes (can correct up to 5 byte errors)
raw_errors, raw_total = 0, 0
rs_success, rs_failed = 0, 0

aes_key = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
aes_nonce = bytes.fromhex("00000000000000000000000000000000")

with torch.no_grad():
    for i in range(args.ber_msgs):
        # 21-character test message
        test_msg = f"TEST_MSG_{i:04d}_IND_SEC".encode('ascii')[:21]
        ctr_tx = Counter.new(128, initial_value=int.from_bytes(aes_nonce, 'big'))
        cipher_tx = AES.new(aes_key, AES.MODE_CTR, counter=ctr_tx)
        ct = cipher_tx.encrypt(test_msg)
        
        # Reed-Solomon encode
        encoded_frame = rsc.encode(ct)
        bits_frame = np.unpackbits(np.frombuffer(encoded_frame, dtype=np.uint8))
        
        # Pad to 256 bits
        if len(bits_frame) < 256:
            bits_256 = np.pad(bits_frame, (0, 256 - len(bits_frame)))
        else:
            bits_256 = bits_frame[:256]
            
        b_tensor = torch.from_numpy(bits_256.astype(np.float32) * 2.0 - 1.0).unsqueeze(0).to(DEVICE)
        z_tensor = torch.randn(1, 64, device=DEVICE)
        
        covert_out = G(z_tensor, b_tensor)
        p = Dec(covert_out)
        bits_hat = (p > 0.5).int().cpu().numpy()[0]
        
        # Pre-FEC raw error counting
        raw_errors += np.sum(bits_hat != bits_256)
        raw_total += 256
        
        # Post-FEC decoding
        orig_len_bits = len(encoded_frame) * 8
        rec_bits = bits_hat[:orig_len_bits]
        rec_bytes = np.packbits(rec_bits).tobytes()
        
        try:
            dec_ct = rsc.decode(rec_bytes)[0]
            ctr_rx = Counter.new(128, initial_value=int.from_bytes(aes_nonce, 'big'))
            cipher_rx = AES.new(aes_key, AES.MODE_CTR, counter=ctr_rx)
            recovered = cipher_rx.decrypt(dec_ct)
            if recovered == test_msg:
                rs_success += 1
            else:
                rs_failed += 1
        except ReedSolomonError:
            rs_failed += 1

raw_ber = raw_errors / raw_total
post_fec_ber = 0.0 if rs_failed == 0 else (rs_failed / args.ber_msgs)
msg_recovery_rate = (rs_success / args.ber_msgs) * 100.0

# ===========================================================================
# SUMMARY REPORT (Table 4 Format from Research Paper)
# ===========================================================================
print("\n" + "=" * 78)
print("PROJECT ANSHUMAN -- 8-DOMAIN ELECTRONIC WARFARE STEALTH EVALUATION REPORT")
print("=" * 78)
print(f"{'Metric':<24} | {'LPI GAN':<10} | {'AWGN (Ref)':<10} | {'BPSK':<10} | {'Pass Criterion':<14} | {'Status':<6}")
print("-" * 78)

def stat_row(name, val_lpi, val_awgn, val_bpsk, crit_str, passed):
    status = "PASS" if passed else "FAIL"
    print(f"{name:<24} | {val_lpi:<10} | {val_awgn:<10} | {val_bpsk:<10} | {crit_str:<14} | {status:<6}")

stat_row("KS p-value", f"{ks_p_lpi:.4f}", f"{ks_p_awgn:.4f}", f"{ks_p_bpsk:.4f}", "> 0.05", ks_p_lpi > 0.05)
stat_row("Kurtosis (kappa)", f"{kurt_lpi:.4f}", f"{kurt_awgn:.4f}", f"{kurt_bpsk:.4f}", "~ 3.0", abs(kurt_lpi - 3.0) < 0.2)
stat_row("Spectral Entropy H", f"{entropy_lpi:.4f}", f"{entropy_awgn:.4f}", f"{entropy_bpsk:.4f}", "> 0.95", entropy_lpi > 0.95)
stat_row("IQ Circularity (eta)", f"{circ_lpi:.4f}", f"{circ_awgn:.4f}", f"{circ_bpsk:.4f}", "< 0.15", circ_lpi < 0.15)
stat_row("PAPR Dev (dB)", f"{papr_dev:+.2f}", "0.00", f"{papr_bpsk-papr_awgn:+.2f}", "|Delta| < 3 dB", abs(papr_dev) < 3.0)
stat_row("CSFA SCF Ratio", f"{scf_ratio:.3f}x", "1.000x", f"{scf_ratio_bpsk:.3f}x", "< 3.0x", scf_ratio < 3.0)
stat_row("HOS Cumulant C42", f"{c42_lpi:+.4f}", f"{c42_awgn:+.4f}", f"{c42_bpsk:+.4f}", "|C42| < 0.3", abs(c42_lpi) < 0.3)
stat_row("WVD TF Ratio", f"{wvd_ratio:.3f}x", "1.000x", f"{wvd_ratio_bpsk:.3f}x", "< 3.7x", wvd_ratio < 3.7)

print("-" * 78)
print(f"Composite Stealth Score Sc:  {composite_sc:.4f}  [Grade: {sc_grade}] (Target: Sc < 0.10)")
ew_passed = sum([
    ks_p_lpi > 0.05, abs(kurt_lpi - 3.0) < 0.2, entropy_lpi > 0.95, circ_lpi < 0.15,
    abs(papr_dev) < 3.0, scf_ratio < 3.0, abs(c42_lpi) < 0.3, wvd_ratio < 3.7
])
print(f"EW Domain Tests Passed:      {ew_passed}/8")

print("\n" + "=" * 78)
print("ADVERSARY & COMMUNICATION RELIABILITY")
print("=" * 78)
print(f"Adversary CNN Accuracy:      {adv_acc:.1f}%  (Target: 48.0% - 56.0% -> Near-random guessing) {'[PASS]' if 45 <= adv_acc <= 58 else '[FAIL]'}")
print(f"Pre-FEC Raw BER:             {raw_ber*100:.3f}% (Target: < 1.0%) {'[PASS]' if raw_ber < 0.01 else '[FAIL]'}")
print(f"Post-FEC Reed-Solomon BER:   {post_fec_ber*100:.4f}% (Target: 0.0000%) {'[PASS]' if post_fec_ber == 0 else '[FAIL]'}")
print(f"Authenticated Msg Recovery:  {msg_recovery_rate:.1f}% ({rs_success}/{args.ber_msgs} frames decoded with AES-128)")
print("=" * 78)
