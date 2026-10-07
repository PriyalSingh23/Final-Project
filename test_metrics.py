#!/usr/bin/env python3
"""Evaluate an LPI-CGAN checkpoint and optionally write a JSON report.

This offline battery measures generated marginal Gaussianity (one-sample KS),
a small CNN's generated-vs-AWGN detection accuracy, a simple autocorrelation
ratio, and noiseless generator-to-decoder BER. It is a model diagnostic, not
proof of covertness over an SDR or in a real RF channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from models import Decoder, GlobalDecoder, Generator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="lpi_checkpoint.pt")
    parser.add_argument("--n-test", type=int, default=2000, help="frames for the KS and autocorrelation diagnostics")
    parser.add_argument("--adv-train", type=int, default=1500)
    parser.add_argument("--adv-epochs", type=int, default=4)
    parser.add_argument("--adv-test", type=int, default=1000)
    parser.add_argument("--ber-msgs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json-out", default=None, help="optional path for machine-readable results")
    args = parser.parse_args()

    if min(args.n_test, args.adv_train, args.adv_epochs, args.adv_test, args.ber_msgs) < 1:
        parser.error("all sample counts and epoch counts must be positive")

    torch.set_num_threads(max(1, args.threads))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_argument = Path(args.ckpt).expanduser()
    checkpoint_path = checkpoint_argument.resolve()
    checkpoint_label = checkpoint_argument.as_posix() if not checkpoint_argument.is_absolute() else str(checkpoint_path)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was added.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or not {"generator", "decoder"} <= checkpoint.keys():
        parser.error("checkpoint must contain 'generator' and 'decoder' state dictionaries")

    generator = Generator(msg_len=256, z_dim=64, out_len=1024).to(device).eval()
    decoder_class = GlobalDecoder if checkpoint.get("decoder_arch") == "global_mlp_v1" else Decoder
    decoder = decoder_class(msg_len=256).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    decoder.load_state_dict(checkpoint["decoder"])
    print(f"Checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"Device: {device} | deterministic seed: {args.seed}")

    @torch.inference_mode()
    def generate(count: int) -> np.ndarray:
        pieces = []
        for start in range(0, count, 250):
            size = min(250, count - start)
            bits = (torch.rand(size, 256, device=device) > 0.5).float().mul(2).sub(1)
            z = torch.randn(size, 64, device=device)
            pieces.append(generator(z, bits).cpu().numpy())
        return np.concatenate(pieces, axis=0)

    print("=" * 64)
    print("1. KOLMOGOROV-SMIRNOV MARGINAL TEST (standard threshold p > 0.05)")
    print("   The README's p > 0.85 is a stricter aspirational target.")
    print("=" * 64)
    generated = generate(args.n_test)
    pooled = generated.reshape(-1)
    mean = float(pooled.mean())
    std = float(pooled.std())
    ks_statistic, ks_pvalue = stats.kstest(pooled, "norm", args=(0, 1))
    ks_statistic = float(ks_statistic)
    ks_pvalue = float(ks_pvalue)
    print(f"pooled N={len(pooled)} mean={mean:.4f} std={std:.4f}")
    print(f"KS statistic: {ks_statistic:.6f}")
    print(f"KS p-value: {ks_pvalue:.4f} {'PASS' if ks_pvalue > 0.05 else 'FAIL'}")

    print("\n" + "=" * 64)
    print("2. AUTOCORRELATION DIAGNOSTIC (not a formal cyclostationarity test)")
    print("=" * 64)
    gen_iq = generated[0]
    generated_complex = gen_iq[0] + 1j * gen_iq[1]
    awgn_complex = (np.random.randn(512) + 1j * np.random.randn(512)) / np.sqrt(2)
    gen_corr = [
        abs(np.mean(generated_complex[:-lag] * np.conj(generated_complex[lag:])))
        for lag in range(1, 21)
    ]
    awgn_corr = [
        abs(np.mean(awgn_complex[:-lag] * np.conj(awgn_complex[lag:])))
        for lag in range(1, 21)
    ]
    cyclo_ratio = float(np.mean(gen_corr) / max(np.mean(awgn_corr), 1e-12))
    print(
        f"Gen mean |Rxx|: {np.mean(gen_corr):.6f} | AWGN: {np.mean(awgn_corr):.6f} "
        f"| ratio {cyclo_ratio:.2f}x"
    )

    print("\n" + "=" * 64)
    print("3. ADVERSARIAL CNN (target 48-56%; one trained detector only)")
    print("=" * 64)

    class AdversaryCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(2, 32, 7, padding=3), torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, 4, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.Conv1d(64, 128, 4, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
                torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 1), torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    adversary = AdversaryCNN().to(device)
    adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=1e-4)
    n_fake_train = (args.adv_train + 1) // 2
    n_real_train = args.adv_train - n_fake_train
    fake_train = torch.from_numpy(generate(n_fake_train)).float()
    real_train = torch.randn(n_real_train, 2, 512)
    train_x = torch.cat((fake_train, real_train), dim=0).to(device)
    train_y = torch.cat((
        torch.ones(n_fake_train, 1), torch.zeros(n_real_train, 1)
    ), dim=0).to(device)
    bce = torch.nn.BCELoss()
    for _ in range(args.adv_epochs):
        order = torch.randperm(args.adv_train, device=device)
        for start in range(0, args.adv_train, 64):
            indices = order[start:start + 64]
            prediction = adversary(train_x[indices])
            loss = bce(prediction, train_y[indices])
            adversary_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adversary_optimizer.step()

    adversary.eval()
    n_fake_test = (args.adv_test + 1) // 2
    n_real_test = args.adv_test - n_fake_test
    fake_test = torch.from_numpy(generate(n_fake_test)).float()
    real_test = torch.randn(n_real_test, 2, 512)
    test_x = torch.cat((fake_test, real_test), dim=0).to(device)
    test_y = torch.cat((
        torch.ones(n_fake_test, 1), torch.zeros(n_real_test, 1)
    ), dim=0).to(device)
    with torch.inference_mode():
        predicted = adversary(test_x).ge(0.5)
        correct = predicted.eq(test_y.bool()).sum().item()
    adversary_accuracy = correct / args.adv_test * 100.0
    print(
        f"Adversary accuracy: {adversary_accuracy:.1f}% "
        f"{'PASS' if 48.0 <= adversary_accuracy <= 56.0 else 'FAIL'}"
    )

    print("\n" + "=" * 64)
    print("4. NOISELESS MODEL-TO-MODEL BER (no channel/SDR impairments)")
    print("=" * 64)
    errors = 0
    total_bits = 0
    with torch.inference_mode():
        for start in range(0, args.ber_msgs, 64):
            size = min(64, args.ber_msgs - start)
            bits = (torch.rand(size, 256, device=device) > 0.5).float().mul(2).sub(1)
            z = torch.randn(size, 64, device=device)
            probabilities = decoder(generator(z, bits))
            predicted = probabilities.ge(0.5)
            errors += predicted.ne(((bits + 1) / 2).bool()).sum().item()
            total_bits += bits.numel()
    ber = errors / max(total_bits, 1)
    print(f"BER: {ber * 100:.2f}% {'PASS' if ber < 0.01 else 'FAIL'}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_label,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "decoder_arch": checkpoint.get("decoder_arch", "conv_v1"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "seed": args.seed,
        "measurement_scope": "synthetic offline model diagnostics; no RF/SDR channel",
        "metrics": {
            "ks_pvalue": ks_pvalue,
            "ks_statistic": ks_statistic,
            "pooled_mean": mean,
            "pooled_std": std,
            "ks_sample_values": int(len(pooled)),
            "cyclo_ratio": cyclo_ratio,
            "adversary_accuracy": adversary_accuracy,
            "adversary_accuracy_pct": adversary_accuracy,
            "ber": float(ber),
            "ber_pct": float(ber * 100.0),
            "ber_bits": int(total_bits),
        },
        "targets": {
            "ks_pvalue_min": 0.05,
            "ks_pvalue_aspirational": 0.85,
            "adversary_accuracy_pct_range": [48.0, 56.0],
            "ber_max": 0.01,
        },
        "passed": {
            "ks_pvalue": bool(ks_pvalue > 0.05),
            "adversary": bool(48.0 <= adversary_accuracy <= 56.0),
            "ber": bool(ber < 0.01),
        },
    }
    report["passed"]["all"] = all(report["passed"].values())

    if args.json_out:
        output = Path(args.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written: {output.resolve()}")


if __name__ == "__main__":
    main()
