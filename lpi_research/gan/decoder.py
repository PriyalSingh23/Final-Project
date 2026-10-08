"""Convolutional Receiver Decoder for LPI-CGAN messages.
Recovers 256 message bits from 512 complex baseband samples.
"""

import torch
import torch.nn as nn


def _init_conv(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Decoder(nn.Module):
    """Receiver Decoder network:
    Input:
      x: [batch, 2, 512] Baseband I/Q samples
    Output:
      probs: [batch, 256] Probabilities for 256 message bits in [0, 1].
    """

    def __init__(self, msg_len: int = 256):
        super().__init__()
        self.msg_len = msg_len
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),   # 512 -> 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 1, 1),                          # (B, 1, 256)
        )
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-channel zero-mean and unit variance normalization for RF robustness
        m = x.mean(dim=2, keepdim=True)
        s = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - m) / s
        logits = self.net(x_norm).squeeze(1)              # (B, 256)
        return torch.sigmoid(logits)

    def hard_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns hard bit decisions: 0 or 1."""
        return (self.forward(x) > 0.5).float()
