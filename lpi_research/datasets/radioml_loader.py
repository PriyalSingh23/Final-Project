"""RadioML & HDF5 Dataset Loader with Train/Validation/Test splitting.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class RadioMLDataset(Dataset):
    """PyTorch Dataset wrapper for multi-modulation IQ data (N, 2, L)."""

    def __init__(self, x_data: np.ndarray, y_labels: np.ndarray, snrs: np.ndarray):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_labels).long()
        self.snrs = torch.from_numpy(snrs).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.snrs[idx]


def load_hdf5_dataset(filepath: str,
                      train_ratio: float = 0.70,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      seed: int = 123) -> Tuple[RadioMLDataset, RadioMLDataset, RadioMLDataset, Dict[str, int]]:
    """Loads an HDF5 dataset, maps labels to integers, and creates stratified splits."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    with h5py.File(filepath, "r") as f:
        x_data = f["X"][:]
        labels_raw = f["labels"][:]
        snrs = f["SNR"][:]

    # Decode string labels
    str_labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels_raw]
    unique_mods = sorted(list(set(str_labels)))
    mod_to_idx = {mod: i for i, mod in enumerate(unique_mods)}
    y_data = np.array([mod_to_idx[l] for l in str_labels], dtype=np.int64)

    # Stratified shuffle & split
    np.random.seed(seed)
    n_total = len(x_data)
    indices = np.random.permutation(n_total)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = RadioMLDataset(x_data[train_idx], y_data[train_idx], snrs[train_idx])
    val_ds = RadioMLDataset(x_data[val_idx], y_data[val_idx], snrs[val_idx])
    test_ds = RadioMLDataset(x_data[test_idx], y_data[test_idx], snrs[test_idx])

    return train_ds, val_ds, test_ds, mod_to_idx
