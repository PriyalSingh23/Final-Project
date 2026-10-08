"""Adversarial Discriminator for LPI Waveforms.
Evaluates time-domain and frequency-domain characteristics against AWGN.
"""

import torch
import torch.nn as nn


def _init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _init_conv(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """Discriminator network:
    Input:
      x: [batch, 2, 512] Baseband I/Q samples
    Output:
      validity: [batch, 1] Real/Fake score (higher -> Real AWGN)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_conv)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)
