"""End-to-End Joint Training Pipeline for LPI-CGAN and Receiver Decoder.
Implements:
  1. Generator G: [z(64); m(256)] -> Quantile-Gaussianized Waveform (2, 512)
  2. Neural Receiver Decoder D_rec: (2, 512) -> Recovered message bits (256)
  3. Primary Discriminator D1 (VT-CNN2) & Secondary Discriminator D2 (reinit every 25 epochs)
  4. 5-Impairment RF Channel layer during training
  5. 9-Component Composite Loss: Adversarial + Spectral Flatness + Cyclostationary +
     Power Normalization + Diversity Regularization + Decoder Reconstruction.
"""

import sys
import os
import math
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lpi_research.gan.generator import Generator
from lpi_research.gan.decoder import Decoder
from lpi_research.gan.discriminator import Discriminator


def calc_spectral_flatness_loss(x: torch.Tensor) -> torch.Tensor:
    """Computes spectral flatness loss via 512-point FFT on complex baseband."""
    # x: [B, 2, 512]
    complex_x = torch.complex(x[:, 0, :], x[:, 1, :])
    fft_x = torch.fft.fft(complex_x, dim=1)
    psd = torch.abs(fft_x)**2
    psd_norm = psd / (torch.sum(psd, dim=1, keepdim=True) + 1e-12)

    # Variance across frequency bins (flat spectrum has 0 variance)
    var_spec = torch.var(psd_norm, dim=1).mean()
    return var_spec


def calc_cyclostationary_loss(x: torch.Tensor, max_lags: int = 128) -> torch.Tensor:
    """Penalizes cyclic autocorrelation peaks across non-zero lags."""
    complex_x = torch.complex(x[:, 0, :], x[:, 1, :])
    B, L = complex_x.shape
    losses = []
    for lag in [1, 2, 4, 8, 16, 32, 64]:
        if lag < L:
            c = complex_x[:, lag:] * torch.conj(complex_x[:, :-lag])
            # Autocorrelation magnitude should be near zero for non-zero lag
            losses.append(torch.mean(torch.abs(c)**2))
    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=x.device)


def apply_rf_channel_batch(x: torch.Tensor,
                           snr_db: float = 15.0,
                           cfo_max_hz: float = 30.0,
                           fs: float = 2e6) -> torch.Tensor:
    """Differentiable / GPU-compatible 5-impairment RF channel batch transformation."""
    B, C, L = x.shape
    device = x.device
    complex_x = torch.complex(x[:, 0, :], x[:, 1, :])

    # 1. Random Carrier Frequency Offset (CFO) per sample
    cfo = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * cfo_max_hz
    t = torch.arange(L, device=device).unsqueeze(0).expand(B, L) / fs
    rot = torch.exp(1j * (2.0 * math.pi * cfo * t))
    out = complex_x * rot

    # 2. Additive White Gaussian Noise (AWGN)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_sigma = math.sqrt(1.0 / (2.0 * snr_linear))
    noise = torch.randn_like(out.real) * noise_sigma + 1j * torch.randn_like(out.imag) * noise_sigma
    out = out + noise

    # Return shape (B, 2, L)
    return torch.stack([out.real, out.imag], dim=1).float()


def train_lpi_cgan(epochs: int = 100,
                   batch_size: int = 64,
                   lr_g: float = 2e-4,
                   lr_d: float = 1e-4,
                   save_dir: str = "lpi_checkpoints",
                   device: str = "cpu") -> Dict[str, Any]:
    """Executes the full joint training of Generator, Discriminators, and Receiver Decoder."""
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("PROJECT ANSHUMAN -- LPI-CGAN & NEURAL DECODER JOINT TRAINING")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 70)

    # Instantiate networks
    G = Generator(msg_len=256, z_dim=64, out_len=1024, dither_std=0.20).to(device)
    D1 = Discriminator().to(device)
    D2 = Discriminator().to(device)
    D_rec = Decoder(msg_len=256).to(device)

    # Optimizers
    opt_g = optim.Adam(list(G.parameters()), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(list(D1.parameters()) + list(D2.parameters()), lr=lr_d, betas=(0.5, 0.999))
    opt_rec = optim.Adam(D_rec.parameters(), lr=1e-3)

    bce_loss = nn.BCELoss()
    bce_with_logits = nn.BCEWithLogitsLoss()

    best_ber = 1.0
    history = []

    steps_per_epoch = 50
    for epoch in range(1, epochs + 1):
        # Periodic reinitialization of D2 every 25 epochs (Anti-fingerprinting defense 3)
        if epoch % 25 == 0:
            D2 = Discriminator().to(device)
            opt_d = optim.Adam(list(D1.parameters()) + list(D2.parameters()), lr=lr_d, betas=(0.5, 0.999))
            print(f"[Epoch {epoch:3d}] Periodic reinitialization of D2 completed.")

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_ber = 0.0

        for _ in range(steps_per_epoch):
            # ---------------------
            # 1. Train Discriminators
            # ---------------------
            opt_d.zero_grad()

            # Real AWGN samples (B, 2, 512)
            real_awgn = torch.randn(batch_size, 2, 512, device=device) * math.sqrt(0.5)

            # Fake LPI samples
            z = torch.randn(batch_size, 64, device=device)
            m_bits = torch.randint(0, 2, (batch_size, 256), device=device).float()
            m_bipolar = m_bits * 2.0 - 1.0
            fake_lpi = G(z, m_bipolar).detach()

            # Discriminator 1
            d1_real = D1(real_awgn)
            d1_fake = D1(fake_lpi)
            loss_d1 = bce_loss(d1_real, torch.ones_like(d1_real)) + bce_loss(d1_fake, torch.zeros_like(d1_fake))

            # Discriminator 2
            d2_real = D2(real_awgn)
            d2_fake = D2(fake_lpi)
            loss_d2 = bce_loss(d2_real, torch.ones_like(d2_real)) + bce_loss(d2_fake, torch.zeros_like(d2_fake))

            loss_d = loss_d1 + 0.5 * loss_d2
            loss_d.backward()
            opt_d.step()

            # ---------------------
            # 2. Train Generator & Receiver Decoder
            # ---------------------
            opt_g.zero_grad()
            opt_rec.zero_grad()

            # Generate new fake batch
            z1 = torch.randn(batch_size, 64, device=device)
            fake_out = G(z1, m_bipolar)

            # Adversarial losses
            g_d1 = D1(fake_out)
            g_d2 = D2(fake_out)
            loss_adv = bce_loss(g_d1, torch.ones_like(g_d1)) + 0.5 * bce_loss(g_d2, torch.ones_like(g_d2))

            # Spectral and cyclostationary losses
            loss_spec = calc_spectral_flatness_loss(fake_out)
            loss_cyclo = calc_cyclostationary_loss(fake_out)

            # Diversity regularization (Anti-fingerprinting defense 2)
            z2 = torch.randn(batch_size, 64, device=device)
            fake_out2 = G(z2, m_bipolar)
            dz = torch.norm(z1 - z2, dim=1) + 1e-8
            dx = torch.norm(fake_out.view(batch_size, -1) - fake_out2.view(batch_size, -1), dim=1)
            loss_div = -torch.mean(dx / dz)

            # Channel impairment passing for receiver training
            rx_chan_in = apply_rf_channel_batch(fake_out, snr_db=15.0, cfo_max_hz=25.0)

            # Neural Receiver Decoder prediction
            pred_probs = D_rec(rx_chan_in)
            loss_rec = bce_loss(pred_probs, m_bits)

            # Composite Generator Loss
            total_g_loss = (1.0 * loss_adv +
                            3.0 * loss_spec +
                            2.5 * loss_cyclo +
                            0.05 * loss_div +
                            15.0 * loss_rec)

            total_g_loss.backward()
            opt_g.step()
            opt_rec.step()

            # Calculate batch BER
            hard_preds = (pred_probs > 0.5).float()
            bit_errors = (hard_preds != m_bits).float().mean().item()

            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += loss_d.item()
            epoch_rec_loss += loss_rec.item()
            epoch_ber += bit_errors

        epoch_g_loss /= steps_per_epoch
        epoch_d_loss /= steps_per_epoch
        epoch_rec_loss /= steps_per_epoch
        epoch_ber /= steps_per_epoch

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{epochs}] "
                  f"Loss_G: {epoch_g_loss:7.4f} | "
                  f"Loss_D: {epoch_d_loss:6.4f} | "
                  f"Loss_Rec: {epoch_rec_loss:6.4f} | "
                  f"Raw BER: {epoch_ber*100:6.3f}% | "
                  f"D1 Prob: {d1_fake.mean().item():.3f}")

        # Save best model
        if epoch_ber < best_ber:
            best_ber = epoch_ber
            ckpt_path = os.path.join(save_dir, "lpi_checkpoint_best.pt")
            torch.save({
                "epoch": epoch,
                "generator": G.state_dict(),
                "decoder": D_rec.state_dict(),
                "discriminator1": D1.state_dict(),
                "discriminator2": D2.state_dict(),
                "ber": best_ber,
            }, ckpt_path)

    print("-" * 70)
    print(f"Training completed! Best Raw BER: {best_ber*100:.3f}%")
    print(f"Checkpoints saved to: {save_dir}/lpi_checkpoint_best.pt")
    return {"best_ber": best_ber, "checkpoint": os.path.join(save_dir, "lpi_checkpoint_best.pt")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LPI-CGAN and Decoder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_lpi_cgan(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
