#!/usr/bin/env python3
"""
RadioML 2018.01A Dataset Loader
Download: https://www.deepsig.ai/datasets  (RML2018.01A.pkl ~2.5 GB)
"""
import os, pickle, numpy as np, requests

RADIOML_URL = "https://opendata.deepsig.io/datasets/2018.01/RML2018.01A.pkl"

def download_radioml(save_path="RML2018.01A.pkl"):
    if os.path.exists(save_path):
        print(f"Dataset exists: {save_path}")
        return save_path
    print(f"Downloading RadioML 2018.01A (~2.5 GB)...")
    response = requests.get(RADIOML_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    print(f"\rDownloaded: {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB", end="")
    print("\nDownload complete.")
    return save_path

def load_radioml(path="RML2018.01A.pkl", target_len=512):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    all_samples, all_labels = [], []
    for (mod_type, snr_db), arr in data.items():
        N = arr.shape[0]
        for i in range(N):
            iq = arr[i]  # (2, 1024)
            all_samples.append(iq[:, :target_len])
            if target_len <= 768:
                all_samples.append(iq[:, 256:256+target_len])
            all_samples.append(iq[:, -target_len:])
            all_labels.extend([(mod_type, snr_db)] * (3 if target_len <= 768 else 2))
    samples = np.array(all_samples, dtype=np.float32)
    for i in range(len(samples)):
        std = samples[i].std()
        if std > 0:
            samples[i] = samples[i] / std
    print(f"Loaded {len(samples)} samples | Mods: {len(set(l[0] for l in all_labels))} | SNR: {min(l[1] for l in all_labels)} to {max(l[1] for l in all_labels)} dB")
    return samples, all_labels

def create_mixed_dataset(radioml_path="RML2018.01A.pkl", n_awgn=25000, target_len=512):
    awgn_samples = np.random.randn(n_awgn, 2, target_len).astype(np.float32)
    if os.path.exists(radioml_path):
        radioml_samples, _ = load_radioml(radioml_path, target_len)
        if len(radioml_samples) > n_awgn:
            indices = np.random.choice(len(radioml_samples), n_awgn, replace=False)
            radioml_samples = radioml_samples[indices]
    else:
        print("RadioML not found. Using pure AWGN.")
        radioml_samples = np.random.randn(n_awgn, 2, target_len).astype(np.float32)
    mixed = np.concatenate([awgn_samples, radioml_samples], axis=0)
    np.random.shuffle(mixed)
    print(f"Mixed dataset: {len(mixed)} total")
    return mixed

if __name__ == '__main__':
    samples, labels = load_radioml("RML2018.01A.pkl")
    print(f"Shape: {samples.shape}")
