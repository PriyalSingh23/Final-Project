"""RadioML VT-CNN2 Electronic Warfare Interception Classifier.
Used to evaluate whether a hostile EW receiver can distinguish LPI waveforms from AWGN.
Target: Classification accuracy ~ 50.0% (random guess -> non-interceptible).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class RadioML_VTCnn2(nn.Module):
    """VT-CNN2 Architecture (O'Shea et al. / RadioML standard).
    Input: [batch, 2, 512] I/Q samples
    Output: [batch, 2] Logits (Class 0: AWGN, Class 1: Signal)
    """

    def __init__(self, in_channels: int = 2, seq_len: int = 512, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 16, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(16 * seq_len, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        h = self.dropout(self.relu(self.fc1(h)))
        out = self.fc2(h)
        return out


def train_and_evaluate_adversary(lpi_samples: np.ndarray,
                                 awgn_samples: np.ndarray,
                                 epochs: int = 5,
                                 batch_size: int = 64,
                                 device: str = "cpu") -> Tuple[float, RadioML_VTCnn2]:
    """Trains an adversary VT-CNN2 to differentiate LPI from AWGN and returns test accuracy.
    Args:
        lpi_samples: [N, 2, 512] float array
        awgn_samples: [N, 2, 512] float array
    Returns:
        (test_accuracy_pct, trained_model)
    """
    N = min(len(lpi_samples), len(awgn_samples))
    x_data = np.concatenate([awgn_samples[:N], lpi_samples[:N]], axis=0).astype(np.float32)
    y_data = np.concatenate([np.zeros(N, dtype=np.int64), np.ones(N, dtype=np.int64)])

    # Shuffle
    perm = np.random.permutation(len(x_data))
    x_data = x_data[perm]
    y_data = y_data[perm]

    # Split train/test (80/20)
    split = int(0.8 * len(x_data))
    x_train, x_test = torch.tensor(x_data[:split]), torch.tensor(x_data[split:])
    y_train, y_test = torch.tensor(y_data[:split]), torch.tensor(y_data[split:])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = RadioML_VTCnn2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

    acc = (correct / max(total, 1)) * 100.0
    return acc, model
