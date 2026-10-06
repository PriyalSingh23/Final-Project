#!/usr/bin/env python3
"""Verify LPI-CGAN Evasion Performance (fixed)

Fixes vs. the original:
  - BER: decoder outputs sigmoid probabilities in [0,1]; the old code
    thresholded at 0 and compared against +-1 bits, which is meaningless.
    Now: bits_hat = (p > 0.5), compared against 0/1 bits.
  - KS / adversary sections: generation batched (much faster on CPU).
  - CLI flags to scale the test down for quick checks.
"""
import argparse, numpy as np, torch
from scipy import stats
from models import Generator, Decoder

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='lpi_checkpoint.pt')
ap.add_argument('--n-test', type=int, default=2000, help='samples for KS test')
ap.add_argument('--adv-train', type=int, default=1500)
ap.add_argument('--adv-epochs', type=int, default=4)
ap.add_argument('--adv-test', type=int, default=1000)
ap.add_argument('--ber-msgs', type=int, default=500)
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator().to(DEVICE).eval()
Dec = Decoder().to(DEVICE).eval()
ckpt = torch.load(args.ckpt, map_location=DEVICE)
G.load_state_dict(ckpt['generator'])
Dec.load_state_dict(ckpt['decoder'])
print(f"Checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

@torch.no_grad()
def gen_batch(n):
    b = (torch.rand(n, 256, device=DEVICE) > 0.5).float() * 2 - 1
    z = torch.randn(n, 64, device=DEVICE)
    return G(z, b).cpu().numpy()

print("=" * 60)
print("1. KOLMOGOROV-SMIRNOV TEST (target p > 0.05, want p > 0.85)")
print("=" * 60)
gen_samples = []
for i in range(0, args.n_test, 250):
    gen_samples.append(gen_batch(min(250, args.n_test - i)))
gen_samples = np.concatenate(gen_samples)              # (N, 2, 512)
gen_flat = gen_samples.reshape(-1)
mu, sd = gen_flat.mean(), gen_flat.std()
print(f"pooled N={len(gen_flat)} mean={mu:.4f} std={sd:.4f}")
ks_stat, p_value = stats.kstest(gen_flat, 'norm', args=(0, 1))
print(f"KS p-value: {p_value:.4f} {'PASS' if p_value > 0.05 else 'FAIL'}")

print("\n" + "=" * 60)
print("2. CYCLOSTATIONARY SUPPRESSION")
print("=" * 60)
gen_single = gen_samples[0].flatten()
gen_complex = gen_single[:512] + 1j * gen_single[512:]
corrs = [np.abs(np.mean(gen_complex[:-lag] * np.conj(gen_complex[lag:]))) for lag in range(1, 21)]
awgn_complex = (np.random.randn(512) + 1j * np.random.randn(512)) / np.sqrt(2)
awgn_corrs = [np.abs(np.mean(awgn_complex[:-lag] * np.conj(awgn_complex[lag:]))) for lag in range(1, 21)]
ratio = np.mean(corrs) / np.mean(awgn_corrs)
print(f"Gen mean |R_xx|: {np.mean(corrs):.6f} | AWGN: {np.mean(awgn_corrs):.6f} | ratio {ratio:.1f}x")

print("\n" + "=" * 60)
print("3. ADVERSARIAL CNN (target ~50%)")
print("=" * 60)
class AdversaryCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, 7, padding=3), torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1), torch.nn.Sigmoid())
    def forward(self, x): return self.net(x)

adv = AdversaryCNN().to(DEVICE)
adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-4)
n_train = args.adv_train
X, y = [], []
for i in range(n_train):
    if i % 2 == 0:
        X.append(torch.from_numpy(gen_batch(1)).float()); y.append(1.0)
    else:
        s = torch.randn(1, 2, 512)
        s = s / s.view(1, -1).pow(2).mean().sqrt()
        X.append(s); y.append(0.0)
X = torch.cat(X, dim=0).to(DEVICE)
y = torch.tensor(y).view(-1, 1).to(DEVICE)
for epoch in range(args.adv_epochs):
    perm = torch.randperm(n_train)
    for i in range(0, n_train, 64):
        idx = perm[i:i + 64]
        pred = adv(X[idx])
        loss = torch.nn.BCELoss()(pred, y[idx])
        adv_opt.zero_grad(); loss.backward(); adv_opt.step()

n_test = args.adv_test; correct = 0
with torch.no_grad():
    for i in range(n_test):
        if i % 2 == 0:
            s = torch.from_numpy(gen_batch(1)).float().to(DEVICE); label = 1
        else:
            s = torch.randn(1, 2, 512).to(DEVICE); label = 0
        p = adv(s).item()
        if (1 if p > 0.5 else 0) == label: correct += 1
acc = correct / n_test * 100
print(f"Adversary accuracy: {acc:.1f}% {'PASS' if 48 <= acc <= 56 else 'FAIL'}")

print("\n" + "=" * 60)
print("4. BIT ERROR RATE (target < 1%)")
print("=" * 60)
errors = 0; total = 0
with torch.no_grad():
    for _ in range(args.ber_msgs):
        b = (torch.rand(1, 256, device=DEVICE) > 0.5).float() * 2 - 1
        z = torch.randn(1, 64, device=DEVICE)
        x = G(z, b)
        p = Dec(x)                                   # sigmoid probs in [0,1]
        bits_hat = (p > 0.5).float()                 # 0/1 -- the old test compared to +-1
        errors += (bits_hat != (b + 1) / 2).sum().item()
        total += 256
ber = errors / total
print(f"BER: {ber*100:.2f}% {'PASS' if ber < 0.01 else 'FAIL'}")
