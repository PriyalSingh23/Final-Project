#!/usr/bin/env python3
"""Fine-tune only the CGAN decoder while keeping the trained generator fixed.

The generator's waveform distribution is therefore unchanged; this is the
safe follow-up when the covert-link BER is above target but the generator's
covertness metrics must not be disturbed.

Example:
    python finetune_decoder.py \
        --input lpi_checkpoint_v3_610.pt \
        --output lpi_checkpoint_decoder_ft.pt \
        --steps 2000 --batch 64

This trains on the same synthetic (z, message) channel used by train.py. It
is not a substitute for OTA/channel-aware training or field verification.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from models import Decoder, Generator

MSG_LEN = 256
Z_DIM = 64
FRAME_LEN = 512


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was added.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not {"generator", "decoder"} <= checkpoint.keys():
        raise ValueError(f"{path} is not an LPI-CGAN checkpoint with generator/decoder weights")
    return checkpoint


def sample_batch(batch_size: int, device: torch.device):
    """Return latent noise, bipolar message symbols, and binary labels."""
    labels = torch.randint(0, 2, (batch_size, MSG_LEN), device=device).float()
    symbols = labels.mul(2.0).sub(1.0)
    noise = torch.randn(batch_size, Z_DIM, device=device)
    return noise, symbols, labels


@torch.inference_mode()
def evaluate_ber(
    generator: Generator,
    decoder: Decoder,
    frames: int,
    batch_size: int,
    device: torch.device,
) -> float:
    errors = 0
    total = 0
    remaining = frames
    while remaining:
        size = min(batch_size, remaining)
        z, symbols, labels = sample_batch(size, device)
        waveforms = generator(z, symbols)
        predicted = decoder(waveforms).ge(0.5)
        errors += predicted.ne(labels.bool()).sum().item()
        total += labels.numel()
        remaining -= size
    return errors / max(total, 1)


def atomic_save(checkpoint: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, tmp_name)
        os.replace(tmp_name, output)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("lpi_checkpoint_v3_610.pt"))
    parser.add_argument("--output", type=Path, default=Path("lpi_checkpoint_decoder_ft.pt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-frames", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="resume decoder tuning from --output")
    args = parser.parse_args()

    if args.steps < 1 or args.batch < 1 or args.eval_frames < 1 or args.eval_every < 1:
        parser.error("steps, batch, eval-frames and eval-every must all be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    source = args.output if args.resume else args.input
    if args.resume and not args.output.is_file():
        parser.error(f"--resume needs an existing output checkpoint: {args.output}")
    checkpoint = _load_checkpoint(source, device)

    generator = Generator(msg_len=MSG_LEN, z_dim=Z_DIM, out_len=FRAME_LEN * 2).to(device).eval()
    decoder = Decoder(msg_len=MSG_LEN).to(device)
    generator.load_state_dict(checkpoint["generator"])
    decoder.load_state_dict(checkpoint["decoder"])
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    if args.resume and "decoder_finetune_optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["decoder_finetune_optimizer"])

    baseline = evaluate_ber(generator, decoder.eval(), args.eval_frames, args.batch, device)
    decoder.train()
    print(
        f"[DECODER-FT] device={device} checkpoint_epoch={checkpoint.get('epoch', '?')} "
        f"baseline_BER={baseline * 100:.3f}% generator=frozen",
        flush=True,
    )

    best_ber = baseline
    best_step = int(checkpoint.get("decoder_finetune", {}).get("steps", 0))
    first_step = best_step + 1 if args.resume else 1
    final_step = best_step + args.steps if args.resume else args.steps
    for step in range(first_step, final_step + 1):
        z, symbols, labels = sample_batch(args.batch, device)
        with torch.no_grad():
            waveforms = generator(z, symbols)
        probabilities = decoder(waveforms)
        loss = F.binary_cross_entropy(probabilities, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == final_step:
            decoder.eval()
            val_ber = evaluate_ber(generator, decoder, args.eval_frames, args.batch, device)
            decoder.train()
            improved = val_ber < best_ber
            if improved:
                best_ber = val_ber
                best_step = step
                checkpoint["decoder"] = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }
                checkpoint["decoder_finetune_optimizer"] = optimizer.state_dict()
                checkpoint["decoder_finetune"] = {
                    "steps": step,
                    "best_step": best_step,
                    "validation_ber": best_ber,
                    "validation_frames": args.eval_frames,
                    "learning_rate": args.lr,
                    "seed": args.seed,
                    "generator_frozen": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_save(checkpoint, args.output)
            print(
                f"step {step:05d}/{final_step} loss={loss.item():.4f} "
                f"validation_BER={val_ber * 100:.3f}% "
                f"best={best_ber * 100:.3f}%{' *' if improved else ''}",
                flush=True,
            )

    if not args.output.is_file():
        # Preserve the source checkpoint if no validation step beat baseline.
        checkpoint["decoder_finetune"] = {
            "steps": final_step,
            "best_step": best_step,
            "validation_ber": best_ber,
            "validation_frames": args.eval_frames,
            "learning_rate": args.lr,
            "seed": args.seed,
            "generator_frozen": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save(checkpoint, args.output)
    print(f"[DECODER-FT] best validation BER={best_ber * 100:.3f}% at step {best_step}; saved {args.output}")


if __name__ == "__main__":
    main()
