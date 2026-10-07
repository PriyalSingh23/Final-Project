#!/usr/bin/env python3
"""Export a trained LPI-CGAN checkpoint to the TorchScript runtime artifacts.

The exported generator/decoder are consumed by the web app and GNU Radio
flowgraphs. The manifest records the exact source checkpoint and artifact
hashes so deployments can identify which trained model they loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from models import Decoder, GlobalDecoder, Generator

PROJECT_DIR = Path(__file__).resolve().parent


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_torchscript(module: torch.jit.ScriptModule, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        module.save(temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(document: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="lpi_checkpoint.pt")
    parser.add_argument("--generator-out", default="generator_lpi.pt")
    parser.add_argument("--decoder-out", default="decoder_lpi.pt")
    parser.add_argument("--manifest-out", default="cgan_manifest.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available")
    device = torch.device(device_name)

    checkpoint_path = project_path(args.checkpoint)
    generator_path = project_path(args.generator_out)
    decoder_path = project_path(args.decoder_out)
    manifest_path = project_path(args.manifest_out)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Older PyTorch versions do not accept weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or not {"generator", "decoder"} <= checkpoint.keys():
        parser.error("checkpoint must contain 'generator' and 'decoder' state dictionaries")

    generator = Generator(msg_len=256, z_dim=64, out_len=1024).to(device).eval()
    decoder_class = GlobalDecoder if checkpoint.get("decoder_arch") == "global_mlp_v1" else Decoder
    decoder = decoder_class(msg_len=256).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    decoder.load_state_dict(checkpoint["decoder"])

    # Generator.forward intentionally samples fresh noise. Disable the tracer's
    # repeatability comparison (which would reject valid stochastic graphs),
    # then validate the saved modules at two batch sizes below.
    z_example = torch.randn(1, 64, device=device)
    b_example = torch.randint(0, 2, (1, 256), device=device).float().mul(2).sub(1)
    x_example = torch.randn(1, 2, 512, device=device)
    traced_generator = torch.jit.trace(
        generator, (z_example, b_example), check_trace=False, strict=True
    )
    traced_decoder = torch.jit.trace(decoder, x_example, check_trace=False, strict=True)
    save_torchscript(traced_generator, generator_path)
    save_torchscript(traced_decoder, decoder_path)

    # Verify the serialized artifacts rather than the eager modules only.
    exported_generator = torch.jit.load(str(generator_path), map_location=device).eval()
    exported_decoder = torch.jit.load(str(decoder_path), map_location=device).eval()
    with torch.inference_mode():
        for batch_size in (1, 3):
            z = torch.randn(batch_size, 64, device=device)
            bits = torch.randint(0, 2, (batch_size, 256), device=device).float().mul(2).sub(1)
            waveform = exported_generator(z, bits)
            decoded = exported_decoder(waveform)
            if waveform.shape != (batch_size, 2, 512):
                raise RuntimeError(f"generator export returned unexpected shape {tuple(waveform.shape)}")
            if decoded.shape != (batch_size, 256):
                raise RuntimeError(f"decoder export returned unexpected shape {tuple(decoded.shape)}")
            if not torch.isfinite(waveform).all() or not torch.isfinite(decoded).all():
                raise RuntimeError("exported model returned non-finite values")

    manifest = {
        "format": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR))
        if checkpoint_path.is_relative_to(PROJECT_DIR) else str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "decoder_arch": checkpoint.get("decoder_arch", "conv_v1"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "generator": {
            "path": str(generator_path.relative_to(PROJECT_DIR))
            if generator_path.is_relative_to(PROJECT_DIR) else str(generator_path),
            "sha256": sha256(generator_path),
            "input": {"latent": 64, "message_bits": 256},
            "output": [2, 512],
        },
        "decoder": {
            "path": str(decoder_path.relative_to(PROJECT_DIR))
            if decoder_path.is_relative_to(PROJECT_DIR) else str(decoder_path),
            "sha256": sha256(decoder_path),
            "input": [2, 512],
            "output_bits": 256,
        },
        "validated_batch_sizes": [1, 3],
    }
    save_json(manifest, manifest_path)
    print(f"Checkpoint epoch {checkpoint.get('epoch', '?')} exported for {device}:")
    print(f"  generator: {generator_path}")
    print(f"  decoder:   {decoder_path}")
    print(f"  manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
