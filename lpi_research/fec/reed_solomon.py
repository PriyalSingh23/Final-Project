"""Reed-Solomon Forward Error Correction (FEC) module for LPI communications.
Supports RS(n, k) encoding and decoding with error correction tracking.
"""

from typing import Tuple
from reedsolo import RSCodec, ReedSolomonError


class ReedSolomonEngine:
    """Reed-Solomon codec for forward error correction.
    Default: n=32 bytes (256 bits), k=23 bytes payload (184 bits), nsym=9 parity bytes.
    Can correct up to floor(9/2) = 4 corrupted bytes (32 bit burst errors) per frame.
    """

    def __init__(self, nsym: int = 9):
        self.nsym = nsym
        self.codec = RSCodec(nsym)

    def encode(self, data: bytes) -> bytes:
        """Appends Reed-Solomon parity bytes to data."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Data must be bytes or bytearray")
        return bytes(self.codec.encode(data))

    def decode(self, encoded: bytes) -> Tuple[bytes, int, bool]:
        """Decodes Reed-Solomon codeword.
        Returns:
            (decoded_bytes, num_errors_corrected, success)
        """
        try:
            # rs.decode returns (decoded_message, decoded_codeword, errata_pos)
            res = self.codec.decode(encoded)
            decoded_msg = bytes(res[0])
            errata = res[2] if len(res) > 2 else []
            err_count = len(errata) if errata is not None else 0
            return decoded_msg, err_count, True
        except ReedSolomonError:
            # Uncorrectable errors
            # Return prefix if possible
            raw_payload = encoded[:-self.nsym] if len(encoded) >= self.nsym else b""
            return raw_payload, -1, False
