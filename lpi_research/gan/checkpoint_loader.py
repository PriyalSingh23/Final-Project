"""Helper module to load trained weights for Generator and Decoder.
Searches project locations for checkpoints or TorchScript exports.
"""

import os
from typing import Tuple, Optional
import torch
from .generator import Generator
from .decoder import Decoder


DEFAULT_CHECKPOINT_PATHS = [
    "lpi_checkpoint.pt",
    "../lpi_checkpoint.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/lpi_checkpoint.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/lpi_checkpoint_v3_610.pt",
]

DEFAULT_GENERATOR_TS = [
    "generator_lpi.pt",
    "../generator_lpi.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/generator_lpi.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/grc/generator_lpi.pt",
]

DEFAULT_DECODER_TS = [
    "decoder_lpi.pt",
    "../decoder_lpi.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/decoder_lpi.pt",
    "c:/Users/gspra/OneDrive/Desktop/LPI_CGAN/grc/decoder_lpi.pt",
]


def load_trained_models(device: str = "cpu",
                        checkpoint_path: Optional[str] = None) -> Tuple[Generator, Decoder]:
    """Loads Generator and Decoder with trained weights."""
    g = Generator().to(device)
    d = Decoder().to(device)

    # 1. Check specified path
    candidates = [checkpoint_path] if checkpoint_path else DEFAULT_CHECKPOINT_PATHS
    found = None
    for path in candidates:
        if path and os.path.exists(path):
            found = path
            break

    if found:
        ckpt = torch.load(found, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            g.load_state_dict(ckpt["generator"], strict=False)
            d.load_state_dict(ckpt["decoder"], strict=False)
        elif isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
            g.load_state_dict(ckpt["generator_state_dict"], strict=False)
            d.load_state_dict(ckpt["decoder_state_dict"], strict=False)
        elif isinstance(ckpt, dict) and "g" in ckpt:
            g.load_state_dict(ckpt["g"], strict=False)
            d.load_state_dict(ckpt["d"], strict=False)
        else:
            # Maybe raw state dict
            try:
                g.load_state_dict(ckpt)
            except Exception:
                pass
    g.eval()
    d.eval()
    return g, d


def load_torchscript_models(device: str = "cpu") -> Tuple[Optional[torch.jit.ScriptModule], Optional[torch.jit.ScriptModule]]:
    """Loads TorchScript exported models if available."""
    g_ts, d_ts = None, None
    for path in DEFAULT_GENERATOR_TS:
        if os.path.exists(path):
            try:
                g_ts = torch.jit.load(path, map_location=device).eval()
                break
            except Exception:
                pass

    for path in DEFAULT_DECODER_TS:
        if os.path.exists(path):
            try:
                d_ts = torch.jit.load(path, map_location=device).eval()
                break
            except Exception:
                pass

    return g_ts, d_ts
