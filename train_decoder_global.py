#!/usr/bin/env python3
"""Train a global-receptive-field decoder against a frozen CGAN generator.

Use this when the legacy convolutional decoder reaches a BER floor. The
waveform generator is loaded from a trained checkpoint and never updated, so
its statistical properties remain unchanged.

Example:
    python train_decoder_global.py \
        --input lpi_checkpoint_v3_610.pt \
        --output lpi_checkpoint_global_decoder.pt \
        --steps 4000 --batch 64
"""
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from models import GlobalDecoder, Generator

MSG_LEN = 256
Z_DIM = 64
FRAME_LEN = 512
DECODER_ARCH = "global_mlp_v1"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not {"generator", "decoder"} <= checkpoint.keys():
        raise ValueError(f"{path} is not an LPI-CGAN checkpoint")
    return checkpoint


def sample_batch(batch_size: int, device: torch.device):
    labels = torch.randint(0, 2, (batch_size, MSG_LEN), device=device).float()
    symbols = labels.mul(2).sub(1)
    noise = torch.randn(batch_size, Z_DIM, device=device)
    return noise, symbols, labels


@torch.inference_mode()
def evaluate_ber(generator, decoder, frames: int, batch_size: int, device: torch.device) -> float:
    errors = 0
    total = 0
    remaining = frames
    while remaining:
        size = min(batch_size, remaining)
        noise, symbols, labels = sample_batch(size, device)
        predicted = decoder(generator(noise, symbols)).ge(0.5)
        errors += predicted.ne(labels.bool()).sum().item()
        total += labels.numel()
        remaining -= size
    return errors / max(total, 1)


def atomic_save(checkpoint: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("lpi_checkpoint_v3_610.pt"))
    parser.add_argument("--output", type=Path, default=Path("lpi_checkpoint_global_decoder.pt"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-frames", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="continue from the checkpoint in --output")
    args = parser.parse_args()

    if min(args.steps, args.batch, args.eval_frames, args.eval_every) < 1:
        parser.error("steps, batch, eval-frames and eval-every must all be positive")
    if args.focal_gamma < 0:
        parser.error("focal-gamma must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    if args.resume:
        if not args.output.is_file():
            parser.error(f"--resume needs an existing output checkpoint: {args.output}")
        checkpoint = load_checkpoint(args.output, device)
        if checkpoint.get("decoder_arch") != DECODER_ARCH:
            parser.error(f"--resume checkpoint is not a {DECODER_ARCH} checkpoint")
    else:
        if args.output.exists():
            parser.error(f"output already exists; choose a new path or pass --resume: {args.output}")
        checkpoint = load_checkpoint(args.input, device)

    generator = Generator(msg_len=MSG_LEN, z_dim=Z_DIM, out_len=FRAME_LEN * 2).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    decoder = GlobalDecoder(msg_len=MSG_LEN).to(device)
    if args.resume:
        decoder.load_state_dict(checkpoint["decoder"])
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in decoder.state_dict().items()
    }
    best_checkpoint = dict(checkpoint)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    initial_ber = evaluate_ber(generator, decoder.eval(), args.eval_frames, args.batch, device)
    decoder.train()
    previous_training = checkpoint.get("decoder_training", {})
    previous_best = float(previous_training.get("validation_ber", initial_ber))
    best_ber = min(initial_ber, previous_best) if args.resume else initial_ber
    best_step = int(previous_training.get("best_model_optimizer_step", previous_training.get("best_step", 0))) if args.resume else 0
    completed_steps = int(previous_training.get("optimizer_steps_total", previous_training.get("steps", 0))) if args.resume else 0
    first_step = completed_steps + 1
    final_step = completed_steps + args.steps
    print(
        f"[GLOBAL-DECODER] device={device} source_epoch={checkpoint.get('epoch', '?')} "
        f"initial_BER={initial_ber * 100:.3f}% best_so_far={best_ber * 100:.3f}% "
        f"steps={first_step}-{final_step} generator=frozen",
        flush=True,
    )

    for step in range(first_step, final_step + 1):
        noise, symbols, labels = sample_batch(args.batch, device)
        with torch.no_grad():
            waveforms = generator(noise, symbols)
        probabilities = decoder(waveforms)
        if args.loss == "focal":
            probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
            cross_entropy = F.binary_cross_entropy(probabilities, labels, reduction="none")
            p_correct = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
            loss = ((1.0 - p_correct).pow(args.focal_gamma) * cross_entropy).mean()
        else:
            loss = F.binary_cross_entropy(probabilities, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == final_step:
            decoder.eval()
            validation_ber = evaluate_ber(generator, decoder, args.eval_frames, args.batch, device)
            decoder.train()
            improved = validation_ber < best_ber
            if improved:
                best_ber = validation_ber
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }
                best_checkpoint = dict(checkpoint)
                best_checkpoint["decoder"] = best_state
                best_checkpoint["decoder_arch"] = DECODER_ARCH
                best_checkpoint["decoder_training"] = {
                    "architecture": DECODER_ARCH,
                    "steps": step,
                    "optimizer_steps_total": step,
                    "best_step": best_step,
                    "best_model_optimizer_step": best_step,
                    "validation_ber": best_ber,
                    "validation_frames": args.eval_frames,
                    "learning_rate": args.lr,
                    "loss": args.loss,
                    "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
                    "seed": args.seed,
                    "generator_frozen": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_save(best_checkpoint, args.output)
            print(
                f"step {step:05d}/{final_step} loss={loss.item():.4f} "
                f"validation_BER={validation_ber * 100:.3f}% "
                f"best={best_ber * 100:.3f}%{' *' if improved else ''}",
                flush=True,
            )

    best_checkpoint["decoder_arch"] = DECODER_ARCH
    best_checkpoint["decoder"] = best_state
    metadata = dict(best_checkpoint.get("decoder_training", {}))
    metadata.update({
        "architecture": DECODER_ARCH,
        "steps": final_step,
        "optimizer_steps_total": final_step,
        "best_step": best_step,
        "best_model_optimizer_step": best_step,
        "validation_ber": best_ber,
        "validation_frames": args.eval_frames,
        "learning_rate": args.lr,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "seed": args.seed,
        "generator_frozen": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    best_checkpoint["decoder_training"] = metadata
    atomic_save(best_checkpoint, args.output)
    print(f"[GLOBAL-DECODER] best validation BER={best_ber * 100:.3f}% at step {best_step}; saved {args.output}")


if __name__ == "__main__":
    main()
