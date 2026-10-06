#!/usr/bin/env python3
"""DeepSig GOLD dataset (HDF5) loader for LPI-CGAN reference signals.

The GOLD dataset ships as GOLD_XYZ_OSC.0001_1024.hdf5:
    X : (N, 1024, 2) float32  -- captured IQ (samples, I/Q)
    Y : (N, 24)   int64       -- one-hot labels (satellite IDs)
    Z : (N, 1)    int64       -- capture metadata

Usage in training: real-world (non-AWGN) reference frames are mixed into
the threat-detector training so the generator must evade a detector that
has seen genuine over-the-air signals, not just synthetic noise.

The file is ~21 GB; we never load it whole -- a random subset is read in
sorted chunks (fast contiguous access) and normalized per sample.
"""
import os, glob, numpy as np, h5py

GLOB_PATTERNS = ("*.hdf5", "*/*.hdf5", "*.h5", "*/*.h5")   # covers the common
                                                           # "downloaded as a
                                                           # folder" layout too


def find_hdf5(explicit=None, cwd="."):
    """Locate the GOLD hdf5; returns the file path or None."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for pat in GLOB_PATTERNS:
        hits = [h for h in glob.glob(os.path.join(cwd, pat)) if os.path.isfile(h)]
        hits = sorted(hits, key=os.path.getsize)
        if hits:
            return hits[-1]                      # largest match wins
    return None


def load_gold(path, n_take=25000, target_len=512, seed=1234, batch_chunk=2000):
    """Random n_take frames -> (n_take, 2, target_len) float32, unit-normalized.

    Each 1024-sample capture contributes one target_len window (random start),
    so the mixed dataset is not temporally aligned to any capture boundary.
    """
    print(f"[GOLD] Opening {path} ...")
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        X = f["X"]
        N = X.shape[0]
        idx = rng.choice(N, size=min(n_take, N), replace=False)
        idx.sort()                                # contiguous-ish reads
        out = np.empty((len(idx), 2, target_len), dtype=np.float32)
        for s in range(0, len(idx), batch_chunk):
            sel = idx[s:s + batch_chunk]
            chunk = X[sel]                        # (k, 1024, 2)
            for j in range(len(sel)):
                start = int(rng.integers(0, 1024 - target_len + 1))
                win = chunk[j, start:start + target_len, :]      # (512, 2)
                out[s + j] = win.T                             # -> (2, 512)
    for i in range(len(out)):                     # per-sample center + normalize
        m = out[i].mean()
        v = out[i].std()
        if v > 0:
            out[i] = (out[i] - m) / v
    print(f"[GOLD] loaded {len(out)} frames of shape {out[0].shape} "
          f"(from {N:,} captures)")
    return out


def find_and_load_real(n_real, target_len=512, explicit=None, cwd="."):
    """Try GOLD hdf5 first, then the classic RML2018.01A.pkl; else None."""
    path = find_hdf5(explicit, cwd)
    if path:
        try:
            return load_gold(path, n_take=n_real, target_len=target_len)
        except Exception as e:
            print(f"[GOLD] failed ({e}); falling back.")
    pkl = os.path.join(cwd, "RML2018.01A.pkl")
    if os.path.isfile(pkl):
        from dataset_radioml import load_radioml
        samples, _ = load_radioml(pkl, target_len)
        if len(samples) > n_real:
            keep = np.random.default_rng(0).choice(len(samples), n_real, replace=False)
            samples = samples[keep]
        return samples
    return None


if __name__ == "__main__":
    import sys, time
    p = sys.argv[1] if len(sys.argv) > 1 else find_hdf5()
    t = time.time()
    s = load_gold(p, n_take=2000)
    print(f"smoke test ok in {time.time()-t:.1f}s | mean {s.mean():.4f} "
          f"std {s.std():.4f} min {s.min():.2f} max {s.max():.2f}")
