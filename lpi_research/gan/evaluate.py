"""Evaluation Script for LPI-CGAN across 8 Electronic Warfare Domains.
Computes:
  1. Kolmogorov-Smirnov test against N(0, 1)
  2. Kurtosis
  3. Spectral Entropy
  4. IQ Circularity
  5. PAPR Deviation
  6. Cyclostationary SCF Ratio
  7. Higher-Order Statistics C42
  8. Spectrogram WVD TF Energy Ratio
  9. Composite Stealth Score Sc
  10. RadioML VT-CNN2 Adversary Interception Accuracy
  11. Bit Error Rate (pre-FEC and post-FEC)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
from typing import Dict, Any
import numpy as np
import torch
from scipy import stats
from scipy.signal import spectrogram
from reedsolo import RSCodec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lpi_research.gan.checkpoint_loader import load_trained_models
from lpi_research.gan.adversary_cnn import RadioML_VTCnn2
from lpi_research.waveform.baseline import generate_awgn, generate_bpsk
from lpi_research.crypto.aes_engine import AESEngine


def calc_spectral_entropy(signals):
    psds = np.abs(np.fft.fft(signals, axis=1))**2
    psd_m = np.mean(psds, axis=0)
    p_norm = psd_m / (np.sum(psd_m) + 1e-12)
    return -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm))


def calc_iq_circularity(signals):
    etas = []
    for sig in signals:
        s = sig - np.mean(sig)
        pseudo_cov = np.abs(np.mean(s**2))
        var = np.mean(np.abs(s)**2) + 1e-12
        etas.append(pseudo_cov / var)
    return float(np.mean(etas))


def calc_mean_papr(signals):
    paprs = []
    for sig in signals:
        p_inst = np.abs(sig)**2
        p_peak = np.max(p_inst)
        p_avg = np.mean(p_inst) + 1e-12
        paprs.append(10.0 * np.log10(p_peak / p_avg))
    return float(np.mean(paprs))


def calc_cyclo_mean(signals, max_lag=20):
    corrs = []
    for i in range(min(len(signals), 200)):
        sig = signals[i]
        c = [np.abs(np.mean(sig[:-lag] * np.conj(sig[lag:]))) for lag in range(1, max_lag + 1)]
        corrs.append(np.mean(c))
    return float(np.mean(corrs))


def calc_c42(signals):
    m4 = np.mean(np.abs(signals)**4)
    m2 = np.mean(np.abs(signals)**2)
    ez2 = np.mean(signals**2)
    return float(np.real(m4 - 2.0 * (m2**2) - np.abs(ez2)**2))


def calc_tf_ratio(signals):
    ratios = []
    for i in range(min(len(signals), 50)):
        _, _, s = spectrogram(signals[i], fs=2e6, nperseg=64, noverlap=32, return_onesided=False)
        s_abs = np.abs(s)
        ratios.append(np.max(s_abs) / (np.mean(s_abs) + 1e-12))
    return float(np.mean(ratios))


def run_full_evaluation(n_frames: int = 1000,
                        checkpoint_path: str = None,
                        device: str = "cpu") -> Dict[str, Any]:
    print("=" * 75)
    print("PROJECT ANSHUMAN -- 8-DOMAIN EW & ADVERSARIAL EVALUATION REPORT")
    print(f"Generating {n_frames} frames ({n_frames * 512} complex baseband samples)...")
    print("=" * 75)

    G, D_rec = load_trained_models(device=device, checkpoint_path=checkpoint_path)

    # 1. Synthesize LPI Waveform Frames
    lpi_complex_frames = []
    lpi_tensor_frames = []
    with torch.no_grad():
        for _ in range(n_frames):
            z = torch.randn(1, 64, device=device)
            m = torch.randint(0, 2, (1, 256), device=device).float() * 2.0 - 1.0
            out = G(z, m)
            lpi_tensor_frames.append(out.cpu().numpy()[0])
            c_sig = (out[0, 0].cpu().numpy() + 1j * out[0, 1].cpu().numpy()).astype(np.complex64)
            lpi_complex_frames.append(c_sig)

    z_sig = np.array(lpi_complex_frames)  # (N, 512)
    lpi_tensors = np.array(lpi_tensor_frames)  # (N, 2, 512)

    # Reference signals
    awgn_raw = np.array([generate_awgn(512) for _ in range(n_frames)])
    bpsk_sig = np.array([generate_bpsk(512) for _ in range(n_frames)])

    # Unit-variance flat arrays for KS & Kurtosis
    flat_lpi = lpi_tensors.reshape(-1)
    flat_awgn = np.concatenate([awgn_raw.real.reshape(-1) * np.sqrt(2.0), awgn_raw.imag.reshape(-1) * np.sqrt(2.0)])
    flat_bpsk = np.concatenate([bpsk_sig.real.reshape(-1), bpsk_sig.imag.reshape(-1)])

    # 1. KS Test
    _, ks_p_lpi = stats.kstest(flat_lpi, stats.norm.cdf)
    _, ks_p_awgn = stats.kstest(flat_awgn, stats.norm.cdf)
    _, ks_p_bpsk = stats.kstest(flat_bpsk, stats.norm.cdf)

    # 2. Kurtosis
    kurt_lpi = stats.kurtosis(flat_lpi, fisher=False)
    kurt_awgn = stats.kurtosis(flat_awgn, fisher=False)
    kurt_bpsk = stats.kurtosis(flat_bpsk, fisher=False)

    # 3. Spectral Entropy
    h_lpi = calc_spectral_entropy(z_sig)
    h_awgn = calc_spectral_entropy(awgn_raw)
    h_bpsk = calc_spectral_entropy(bpsk_sig)

    # 4. IQ Circularity
    circ_lpi = calc_iq_circularity(z_sig)
    circ_awgn = calc_iq_circularity(awgn_raw)
    circ_bpsk = calc_iq_circularity(bpsk_sig)

    # 5. PAPR Deviation
    papr_lpi = calc_mean_papr(z_sig)
    papr_awgn = calc_mean_papr(awgn_raw)
    papr_bpsk = calc_mean_papr(bpsk_sig)
    papr_dev = papr_lpi - papr_awgn

    # 6. CSFA SCF Ratio
    cyclo_lpi = calc_cyclo_mean(z_sig)
    cyclo_awgn = calc_cyclo_mean(awgn_raw)
    cyclo_bpsk = calc_cyclo_mean(bpsk_sig)
    scf_ratio = cyclo_lpi / cyclo_awgn
    scf_ratio_bpsk = cyclo_bpsk / cyclo_awgn

    # 7. Higher-Order Statistics C42
    c42_lpi = calc_c42(z_sig)
    c42_awgn = calc_c42(awgn_raw)
    c42_bpsk = calc_c42(bpsk_sig)

    # 8. WVD Time-Frequency Energy Ratio
    tf_lpi = calc_tf_ratio(z_sig)
    tf_awgn = calc_tf_ratio(awgn_raw)
    tf_bpsk = calc_tf_ratio(bpsk_sig)
    wvd_ratio = tf_lpi / tf_awgn
    wvd_ratio_bpsk = tf_bpsk / tf_awgn

    # Composite Stealth Score Sc
    d_ks = max(0.0, 1.0 - ks_p_lpi)
    d_kurt = min(1.0, abs(kurt_lpi - 3.0) / 1.5)
    d_ent = max(0.0, 1.0 - h_lpi)
    d_circ = min(1.0, circ_lpi / 0.5)
    d_papr = min(1.0, abs(papr_dev) / 5.0)
    d_csfa = min(1.0, max(0.0, (scf_ratio - 1.0) / 3.0))
    d_c42 = min(1.0, abs(c42_lpi) / 0.5)
    d_wvd = min(1.0, max(0.0, (wvd_ratio - 1.0) / 3.0))

    sc = (0.20 * d_ks + 0.20 * d_kurt + 0.15 * d_ent + 0.10 * d_circ +
          0.10 * d_papr + 0.10 * d_csfa + 0.10 * d_c42 + 0.05 * d_wvd)
    grade = "Excellent" if sc < 0.10 else "Good" if sc < 0.20 else "Moderate" if sc < 0.35 else "Unacceptable"

    print("\n8-DOMAIN ELECTRONIC WARFARE EVALUATION")
    print("-" * 75)
    print(f"{'Metric':<24} | {'LPI-CGAN':<10} | {'AWGN (Ref)':<10} | {'BPSK':<10} | {'Status'}")
    print("-" * 75)
    print(f"{'1. KS p-value':<24} | {ks_p_lpi:<10.4f} | {ks_p_awgn:<10.4f} | {ks_p_bpsk:<10.4f} | {'PASS' if ks_p_lpi > 0.05 else 'FAIL'}")
    print(f"{'2. Kurtosis (kappa)':<24} | {kurt_lpi:<10.4f} | {kurt_awgn:<10.4f} | {kurt_bpsk:<10.4f} | {'PASS' if abs(kurt_lpi-3.0)<0.2 else 'FAIL'}")
    print(f"{'3. Spectral Entropy (H)':<24} | {h_lpi:<10.4f} | {h_awgn:<10.4f} | {h_bpsk:<10.4f} | {'PASS' if h_lpi > 0.95 else 'FAIL'}")
    print(f"{'4. IQ Circularity (eta)':<24} | {circ_lpi:<10.4f} | {circ_awgn:<10.4f} | {circ_bpsk:<10.4f} | {'PASS' if circ_lpi < 0.15 else 'FAIL'}")
    print(f"{'5. PAPR Deviation (dB)':<24} | {papr_dev:<+10.2f} | {0.00:<10.2f} | {papr_bpsk-papr_awgn:<+10.2f} | {'PASS' if abs(papr_dev)<3.0 else 'FAIL'}")
    print(f"{'6. CSFA SCF Ratio':<24} | {scf_ratio:<10.3f}x| {1.000:<10.3f}x| {scf_ratio_bpsk:<10.3f}x| {'PASS' if scf_ratio < 3.0 else 'FAIL'}")
    print(f"{'7. HOS Cumulant C42':<24} | {c42_lpi:<10.4f} | {c42_awgn:<10.4f} | {c42_bpsk:<10.4f} | {'PASS' if abs(c42_lpi)<0.3 else 'FAIL'}")
    print(f"{'8. WVD TF Ratio':<24} | {wvd_ratio:<10.3f}x| {1.000:<10.3f}x| {wvd_ratio_bpsk:<10.3f}x| {'PASS' if wvd_ratio < 3.7 else 'FAIL'}")
    print("-" * 75)
    print(f"Composite Stealth Score Sc: {sc:.4f} [Grade: {grade}]")

    # 9. Adversary RadioML VT-CNN2 Classifier
    print("\nTraining Adversarial Electronic Warfare CNN (RadioML VT-CNN2)...")
    adv = RadioML_VTCnn2(in_channels=2, seq_len=512, num_classes=2).to(device)
    adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-3)
    adv_crit = torch.nn.CrossEntropyLoss()

    def rand_circular_shift(x):
        B, _, T = x.shape
        shifts = torch.randint(0, T, (B, 1, 1), device=x.device)
        idx = (torch.arange(T, device=x.device).view(1, 1, T) + shifts) % T
        return torch.gather(x, 2, idx.expand(B, x.shape[1], T))

    adv.train()
    for _ in range(5):
        z_b = torch.randn(64, 64, device=device)
        m_b = torch.randint(0, 2, (64, 256), device=device).float() * 2.0 - 1.0
        with torch.no_grad():
            covert_batch = G(z_b, m_b)
            covert_batch = rand_circular_shift(covert_batch)
        noise_batch = torch.randn(64, 2, 512, device=device)
        noise_batch = noise_batch / noise_batch.view(64, -1).pow(2).mean(dim=1, keepdim=True).sqrt().unsqueeze(2)

        x_batch = torch.cat([noise_batch, covert_batch], dim=0)
        y_batch = torch.cat([torch.zeros(64, dtype=torch.long, device=device),
                             torch.ones(64, dtype=torch.long, device=device)])

        adv_opt.zero_grad()
        loss = adv_crit(adv(x_batch), y_batch)
        loss.backward()
        adv_opt.step()

    adv.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(200):
            z_t = torch.randn(1, 64, device=device)
            m_t = torch.randint(0, 2, (1, 256), device=device).float() * 2.0 - 1.0
            out = G(z_t, m_t)
            out = rand_circular_shift(out)
            if adv(out).argmax().item() == 1: correct += 1
            noise = torch.randn(1, 2, 512, device=device)
            noise = noise / noise.view(1, -1).pow(2).mean().sqrt()
            if adv(noise).argmax().item() == 0: correct += 1

    adv_acc = (correct / 400.0) * 100.0
    print(f"Adversary CNN Interception Accuracy: {adv_acc:.1f}% (Target: ~50.0% -> Random Guess)")

    # 10. Communication Reliability & Reed-Solomon Decode
    print("\nCOMMUNICATION RELIABILITY & DECODER METRICS")
    print("-" * 75)
    rsc = RSCodec(9)
    aes = AESEngine()
    raw_errors = 0
    total_bits = 0
    passed_frames = 0
    test_frames = min(n_frames, 200)

    with torch.no_grad():
        for i in range(test_frames):
            test_msg = f"TEST_SEC_{i:03d}".encode('utf-8')
            enc_frame = aes.pack_secure_frame(test_msg, max_payload_len=21)
            rs_encoded = rsc.encode(enc_frame)
            bits_256 = np.unpackbits(np.frombuffer(rs_encoded, dtype=np.uint8))

            b_t = torch.from_numpy(bits_256.astype(np.float32) * 2.0 - 1.0).unsqueeze(0).to(device)
            z_t = torch.randn(1, 64, device=device)
            out_iq = G(z_t, b_t)

            probs = D_rec(out_iq)
            bits_hat = (probs > 0.5).int().cpu().numpy()[0]

            raw_errors += np.sum(bits_hat != bits_256)
            total_bits += 256

            rec_bytes = np.packbits(bits_hat[:len(rs_encoded)*8]).tobytes()
            try:
                dec_bytes = rsc.decode(rec_bytes)[0]
                rec_text, crc_ok = aes.unpack_secure_frame(dec_bytes, max_payload_len=21)
                if crc_ok and rec_text == test_msg:
                    passed_frames += 1
            except Exception:
                pass

    pre_fec_ber = (raw_errors / max(total_bits, 1)) * 100.0
    print(f"Pre-FEC Raw BER:            {pre_fec_ber:.4f}% (Target: < 1.0%)")
    print(f"Post-FEC Decoded:           {passed_frames}/{test_frames} frames ({passed_frames*100/test_frames:.1f}%)")
    print(f"Post-FEC Reed-Solomon BER:  {0.0000 if passed_frames == test_frames else (test_frames-passed_frames)*100/test_frames:.4f}%")
    print("=" * 75)


if __name__ == "__main__":
    import math
    parser = argparse.ArgumentParser(description="Evaluate LPI Waveform")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to evaluate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_full_evaluation(n_frames=args.frames, device=args.device)
