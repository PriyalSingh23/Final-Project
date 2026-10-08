"""Matrix Block Interleaver and Deinterleaver for dispersal of burst channel errors.
"""

import numpy as np


class MatrixInterleaver:
    """Interleaves and deinterleaves 1D bit or byte arrays using an (R x C) matrix.
    Data is written in row-major order and read out in column-major order.
    """

    def __init__(self, rows: int = 16, cols: int = 16):
        self.rows = rows
        self.cols = cols
        self.total_size = rows * cols

    def interleave_bits(self, bits: np.ndarray) -> np.ndarray:
        """Interleaves an array of bits (must match or be padded to rows * cols)."""
        if len(bits) != self.total_size:
            pad = (self.total_size - (len(bits) % self.total_size)) % self.total_size
            bits = np.pad(bits, (0, pad), mode='constant')

        matrix = bits.reshape((self.rows, self.cols))
        # Read out column-major
        return matrix.T.reshape(-1)

    def deinterleave_bits(self, bits: np.ndarray) -> np.ndarray:
        """Deinterleaves an array of bits."""
        if len(bits) != self.total_size:
            raise ValueError(f"Bit length {len(bits)} does not match interleaver size {self.total_size}")

        matrix = bits.reshape((self.cols, self.rows)).T
        return matrix.reshape(-1)

    def interleave_bytes(self, data: bytes) -> bytes:
        """Converts bytes to bits, interleaves, and packs back to bytes."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        interleaved = self.interleave_bits(bits)
        return np.packbits(interleaved).tobytes()

    def deinterleave_bytes(self, data: bytes) -> bytes:
        """Unpacks bytes to bits, deinterleaves, and packs back to bytes."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        deinterleaved = self.deinterleave_bits(bits)
        return np.packbits(deinterleaved).tobytes()
