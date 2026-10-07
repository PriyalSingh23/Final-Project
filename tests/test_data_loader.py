"""Small local tests for optional capture-loader path selection."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_gold import find_and_load_real


class CaptureLoaderTests(unittest.TestCase):
    def test_explicit_radioml_pickle_path_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny_radioml.pkl"
            fixture = {("QPSK", 0): np.ones((1, 2, 1024), dtype=np.float32)}
            with path.open("wb") as stream:
                pickle.dump(fixture, stream)

            samples = find_and_load_real(n_real=2, target_len=512, explicit=str(path))

        self.assertEqual(samples.shape, (2, 2, 512))
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
