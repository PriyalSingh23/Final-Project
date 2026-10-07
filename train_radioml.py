#!/usr/bin/env python3
"""Compatibility wrapper for ``python train.py --radioml``.

The optional GOLD HDF5 or RadioML pickle is used only as additional
real-signal examples for the threat-model detector. It does not replace the
generator's synthetic AWGN spectral reference and does not establish RF or
LPI performance. Provide a dataset with ``--data PATH``; this wrapper does
not download one automatically.

Example:
    python train_radioml.py --data /path/to/captures.hdf5
"""
import sys
import train

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--radioml", *sys.argv[1:]]
    train.main()
