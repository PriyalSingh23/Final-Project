"""Unit tests for synthetic dataset generator and loader.
"""

import os
import tempfile
import numpy as np
from lpi_research.datasets import generate_single_sample, generate_dataset_hdf5, load_hdf5_dataset


def test_generate_single_sample():
    for mod in ['AWGN', 'BPSK', 'QPSK', 'OFDM']:
        sig = generate_single_sample(mod, n_samples=512, snr_db=15.0)
        assert len(sig) == 512
        assert sig.dtype == np.complex64
        # Power should be finite and normalized around 1.0
        pwr = np.mean(np.abs(sig)**2)
        assert 0.5 < pwr < 2.0


def test_generate_and_load_hdf5():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test_dataset.h5")
        generate_dataset_hdf5(h5_path, samples_per_mod=5, n_samples=256, snr_range=[0, 10])
        assert os.path.exists(h5_path)

        train_ds, val_ds, test_ds, mod_map = load_hdf5_dataset(h5_path)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        x, y, snr = train_ds[0]
        assert x.shape == (2, 256)
