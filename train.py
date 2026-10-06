#!/usr/bin/env python3
"""
LPI-CGAN Training (Fixed)
--------------------------
Fixes vs. the original failing version:

1. BIT ERROR RATE (was ~43-50%):
   The Decoder outputs sigmoid probabilities in [0,1], but the old code
   computed F.mse_loss(b_hat, b) with b in {-1,+1}. The optimal decoder
   under that loss is the constant 0.5 -- so the decoder never learned
   the bits. Now we use binary cross-entropy against 0/1 bits, matching
   the decoder's output range.

2. KS TEST (was p = 0.0000):
   Gaussianize eps tightened 1e-3 -> 1e-4. With eps=1e-3 the transform
   clamps at ~3.09 sigma and the pooled KS over ~2M points detects the
   missing tail mass. 1e-4 keeps gradients finite while passing KS.

3. CYCLOSTATIONARITY (was |R_xx| ~1.41 vs AWGN 0.045):
   Cyclo loss weight 1.0 -> 10.0 and lag range 20 -> 64.

4. ADVERSARIAL CNN (was 96.9%):
   Consequence of (2) and (3); stronger suppression of correlation /
   spectral tells.  Discriminator already uses spectral normalization,
   so the WGAN weight-clamp hack is removed.

Other changes:
  - argparse: --epochs --n-samples --batch --out --resume --radioml
  - checkpoint saved EVERY epoch (resume-safe, works with 300 s limits)
  - quick BER monitor printed every epoch
"""
import os, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import Generator, Discriminator, Decoder

# ==================== CONFIG ====================
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
LR_G         = 1e-4
LR_D         = 2e-4
N_CRITIC     = 3          # D steps per G step (D has spectral norm)
D2_RESET_EVERY = 25
FRAME_LEN    = 512
MSG_LEN      = 256
Z_DIM        = 64

W_D2         = 0.5
W_REC        = 40.0
W_MEAN       = 1.0
W_VAR        = 1.0
W_KURT       = 2.0        # ep 607: center-peak sharpening (excess kurt +0.017) is the KS killer now
W_SPEC       = 5.0
W_CYCLO      = 15.0       # was 1.0 -- far too weak
W_XCORR      = 3.0        # ep 614: systematic negative I-Q correlation (-0.012 vs +0.001)
W_DIV        = 0.1
W_QUANT      = 16.0        # quantile match to N(0,1) -- directly optimizes KS
W_DET        = 8.0        # threat-model detector fooling (same arch as test adversary)
W_PSDCV      = 6.0        # periodogram coefficient-of-variation (shape, not just mean)

RADIOML_PATH = 'RML2018.01A.pkl'

# ==================== DATASET ====================
class LPIDataset(Dataset):
    def __init__(self, n=50000, use_radioml=False, data_path=None):
        self.n = n
        # GOLD/RadioML frames are used ONLY as extra "real" examples for the
        # threat-model detector (train_detector). The spectral-shape reference
        # for the generator stays pure AWGN -- otherwise the spectral loss
        # would imprint GNSS line-spectrum structure onto the covert signal.
        self.gold_bank = None
        if use_radioml:
            from dataset_gold import find_and_load_real
            print("Loading real-world reference dataset (GOLD hdf5 / RadioML pkl)...")
            self.gold_bank = find_and_load_real(n_real=max(n // 2, 10000),
                                                target_len=FRAME_LEN, explicit=data_path)
            if self.gold_bank is None:
                print("No real-world dataset found -- detector will see AWGN only.")
        self.real_bank = np.random.randn(n, 2, FRAME_LEN).astype(np.float32)
        self.messages = [torch.randint(0, 2, (MSG_LEN,)).float() * 2.0 - 1.0 for _ in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        b = self.messages[idx]
        real_idx = np.random.randint(0, len(self.real_bank))
        real = torch.from_numpy(self.real_bank[real_idx].copy())
        return b, real

# ==================== AUXILIARY LOSSES ====================
def loss_mean(x):
    return x.mean() ** 2

def loss_var(x):
    B = x.shape[0]
    var = x.view(B, -1).var(dim=1, unbiased=False).mean()
    return (var - 1.0) ** 2

def loss_kurtosis(x):
    x = x.view(-1)
    mu = x.mean()
    sig2 = x.var(unbiased=False)
    kurt = ((x - mu) ** 4).mean() / (sig2 ** 2 + 1e-8)
    return (kurt - 3.0) ** 2

def loss_spectral(gen, ref, n_bands=16):
    gen_psd = torch.abs(torch.fft.rfft(gen, dim=2)) ** 2
    ref_psd = torch.abs(torch.fft.rfft(ref, dim=2)) ** 2
    gen_psd = gen_psd / (gen_psd.sum(dim=2, keepdim=True) + 1e-8)
    ref_psd = ref_psd / (ref_psd.sum(dim=2, keepdim=True) + 1e-8)
    # log domain: a multiplicative tilt (e.g. excess low-frequency energy)
    # is invisible to linear-PSD MSE but jumps out in log-PSD
    loss = F.mse_loss(torch.log(gen_psd + 1e-6), torch.log(ref_psd + 1e-6))
    # coarse per-band energy match -- directly kills band-level tells
    B, C, Fbins = gen_psd.shape
    k = Fbins // n_bands
    gb = gen_psd[:, :, :k * n_bands].view(B, C, n_bands, k).mean(dim=3)
    rb = ref_psd[:, :, :k * n_bands].view(B, C, n_bands, k).mean(dim=3)
    loss = loss + 1.0 * F.mse_loss(gb, rb)
    return loss

def loss_cyclo(x, max_lag=None):
    """Full-lag circular autocorrelation whitening via FFT.

    Long-lag correlation is exactly low-frequency spectral tilt -- the
    signature every fresh detector instance kept re-finding. A temporally
    white signal has a naturally fluctuating (chi^2) periodogram, which
    is what real AWGN looks like."""
    B, C, L = x.shape
    X = torch.fft.rfft(x, dim=2)
    ac = torch.fft.irfft(X * torch.conj(X), n=L, dim=2)  # circular autocorr
    ac = ac[:, :, 1:] / L                                 # lags 1..L-1
    if max_lag is not None:
        ac = ac[:, :, :max_lag]
    # short lags get more weight: smooth structure lives there
    lags = torch.arange(1, ac.shape[2] + 1, device=x.device, dtype=torch.float32)
    w = 1.0 / lags.sqrt()
    return (ac ** 2 * w).sum() / w.sum()

def loss_psd_cv(x):
    """Periodogram of white AWGN has std/mean ~ 1 (exponential values).
    The generator's deterministic outputs smooth it (tell d=-6). Match it."""
    p = torch.abs(torch.fft.rfft(x, dim=2)) ** 2
    cv = p.std(dim=2) / (p.mean(dim=2) + 1e-8)
    return ((cv - 1.0) ** 2).mean()

def loss_iq_xcorr(x):
    I = x[:, 0, :] - x[:, 0, :].mean(dim=1, keepdim=True)
    Q = x[:, 1, :] - x[:, 1, :].mean(dim=1, keepdim=True)
    return (I * Q).mean(dim=1).pow(2).mean()

def loss_diversity(x):
    B = x.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=x.device)
    x_flat = x.view(B, -1)
    dist = torch.cdist(x_flat, x_flat, p=2)
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    return -dist[mask].mean()

_SQRT2 = math.sqrt(2.0)

def _quantile_one(flat, n_pts, device):
    if flat.numel() > n_pts:
        idx = torch.randint(0, flat.numel(), (n_pts,), device=device)
        flat = flat[idx]
    n = flat.numel()
    s, _ = torch.sort(flat)
    pos = (torch.arange(n, device=device, dtype=torch.float32) + 0.375) / (n + 0.25)
    target = _SQRT2 * torch.erfinv(2 * pos - 1)
    return F.mse_loss(s, target)

def loss_ks(x, n_pts=10**9):
    """KS statistic itself: sort values, map through Phi, sup deviation from
    the uniform grid. CDF space (not quantile space), so no region gets an
    outsized weight -- and the per-sample power normalization's capped tails
    (impossible 4-sigma order statistics) don't generate useless gradient."""
    worst = torch.zeros((), device=x.device)
    for c in range(x.shape[1]):
        flat = x[:, c, :].reshape(-1)
        if flat.numel() > n_pts // 2:
            idx = torch.randint(0, flat.numel(), (n_pts // 2,), device=x.device)
            flat = flat[idx]
        n = flat.numel()
        sv, _ = torch.sort(flat)
        u = 0.5 * (1.0 + torch.erf(sv / _SQRT2))          # CDF of sorted values
        pos = (torch.arange(n, device=x.device, dtype=torch.float32) + 0.375) / (n + 0.25)
        worst = torch.maximum(worst, (u - pos).abs().max())
    return worst

def loss_quantile(x, n_pts=65536):
    """Match batch quantiles to standard-normal order statistics.

    Computed PER CHANNEL: the generator otherwise lets I and Q drift to
    different distributions (a mixture the pooled KS detects instantly).
    """
    l_i = _quantile_one(x[:, 0, :].reshape(-1), n_pts // 2, x.device)
    l_q = _quantile_one(x[:, 1, :].reshape(-1), n_pts // 2, x.device)
    return 0.5 * (l_i + l_q)

def loss_iq_match(x):
    """I and Q must share mean/std/kurtosis, like proper AWGN -- AND each
    channel's pooled mean must be 0. At ep 600 the MLP learned an
    ANTISYMMETRIC DC offset (I +0.014 / Q -0.012): invisible to loss_mean
    (overall mean ~0) and to the old (mi-mq)^2 batch term's gradient budget,
    but a per-channel KS sees the shifted marginal instantly."""
    mi = x[:, 0, :].mean(); mq = x[:, 1, :].mean()
    si = x[:, 0, :].std();  sq = x[:, 1, :].std()
    def kurt(c):
        m = c.mean(); v = c.var()
        return (((c - m) ** 4).mean() / (v ** 2 + 1e-8))
    ki = kurt(x[:, 0, :]); kq = kurt(x[:, 1, :])
    return ((mi - mq) ** 2 + mi ** 2 + mq ** 2 +
            (si - sq) ** 2 + 0.1 * (ki - kq) ** 2)


# ==================== THREAT-MODEL DETECTOR ====================
class Detector(nn.Module):
    """Same architecture as the test's adversary CNN. Trained fresh each
    epoch on gen-vs-AWGN; the generator is then penalized for being
    detectable by it. This optimizes exactly what metric 3 measures."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def train_detector(G, real_bank=None, steps=60, batch=64):
    """Threat-model detector. Its 'real' class is half pure AWGN and half
    real-world captures (GOLD) when a bank is available, so the generator
    must evade a detector that knows what genuine over-the-air signals
    look like -- not just synthetic noise."""
    det = Detector().to(DEVICE)
    opt = torch.optim.Adam(det.parameters(), lr=1e-4)
    half = batch // 2
    for _ in range(steps):
        with torch.no_grad():
            b = (torch.rand(batch, MSG_LEN, device=DEVICE) > 0.5).float() * 2 - 1
            z = torch.randn(batch, Z_DIM, device=DEVICE)
            fake = G(z, b)
        real = torch.randn(batch, 2, FRAME_LEN, device=DEVICE)
        if real_bank is not None:
            idx = np.random.randint(0, len(real_bank), half)
            real[:half] = torch.from_numpy(real_bank[idx].copy()).to(DEVICE)
        x = torch.cat([fake, real], 0)
        y = torch.cat([torch.ones(batch, 1), torch.zeros(batch, 1)], 0).to(DEVICE)
        loss = F.binary_cross_entropy(det(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return det.eval()

# ==================== TRAINING ====================

def detector_accuracy(det, G, n=1000):
    det.eval()
    correct = 0
    with torch.no_grad():
        for i in range(n):
            if i % 2 == 0:
                b = (torch.rand(1, MSG_LEN, device=DEVICE) > 0.5).float() * 2 - 1
                z = torch.randn(1, Z_DIM, device=DEVICE)
                s = G(z, b); label = 1
            else:
                s = torch.randn(1, 2, FRAME_LEN, device=DEVICE); label = 0
            if (det(s).item() > 0.5) == bool(label):
                correct += 1
    return correct / n

def quick_ber(G, Dec, n=64):
    """Cheap in-training BER monitor (same convention as test_metrics)."""
    G.eval(); Dec.eval()
    errs, tot = 0, 0
    with torch.no_grad():
        for _ in range(4):
            b = (torch.rand(n, MSG_LEN, device=DEVICE) > 0.5).float() * 2 - 1
            z = torch.randn(n, Z_DIM, device=DEVICE)
            x = G(z, b)
            p = Dec(x)                                   # sigmoid probs in [0,1]
            bits_hat = (p > 0.5).float()
            errs += (bits_hat != (b + 1) / 2).sum().item()
            tot += b.numel()
    G.train(); Dec.train()
    return errs / tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--n-samples', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default='lpi_checkpoint.pt')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--radioml', action='store_true', help='mix real-world reference signals (GOLD hdf5 / RadioML pkl) into detector training')
    ap.add_argument('--data', default=None, help='explicit path to GOLD .hdf5 or RML2018.01A.pkl')
    args = ap.parse_args()

    print(f"[TRAIN] Device: {DEVICE} | samples: {args.n_samples} | epochs: {args.epochs} | RadioML mix: {args.radioml}")

    G     = Generator(msg_len=MSG_LEN, z_dim=Z_DIM, out_len=FRAME_LEN * 2).to(DEVICE)
    Dec   = Decoder(msg_len=MSG_LEN).to(DEVICE)
    # NOTE: the WGAN critics (D1/D2) were removed. Their unbounded linear
    # scores ran away (D1(fake) reached +138), drowning every statistical
    # loss in the generator gradient. The sigmoid-bounded threat-model
    # Detector is the only adversary -- same architecture as the adversary
    # CNN used by test_metrics.py.
    g_opt  = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.9))
    dec_opt = torch.optim.Adam(Dec.parameters(), lr=2e-4, betas=(0.5, 0.9))  # stepped every batch

    start_epoch = 1
    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=DEVICE)
        G.load_state_dict(ckpt['generator']); Dec.load_state_dict(ckpt['decoder'])
        try:
            g_opt.load_state_dict(ckpt['g_opt'])
        except (KeyError, ValueError):
            print("  (optimizer state not compatible -- starting optimizers fresh)")
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {args.out} at epoch {ckpt['epoch']}")

    dataset = LPIDataset(args.n_samples, use_radioml=args.radioml, data_path=args.data)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    det = train_detector(G, dataset.gold_bank, steps=60)  # threat-model detector, refreshed each epoch
    for epoch in range(start_epoch, args.epochs + 1):
        g_losses, rec_losses = [], []

        # anneal invisibility weights in over the first 30 epochs: the bit
        # encoding must establish itself before being hidden, otherwise the
        # statistical losses suppress it and the decoder never locks on
        ramp = min(1.0, epoch / 30.0)

        for batch_idx, (b, real_awgn) in enumerate(loader):
            b = b.to(DEVICE)
            real = real_awgn.to(DEVICE)
            B = b.size(0)

            # Train G + Decoder
            if True:
                z = torch.randn(B, Z_DIM, device=DEVICE)
                fake = G(z, b)
                b_hat = Dec(fake)
                bits = (b + 1) / 2                                   # 0/1 -- matches sigmoid

                g_rec = F.binary_cross_entropy(b_hat, bits)          # was MSE vs +-1: never learned
                g_det = F.binary_cross_entropy(det(fake), torch.zeros(B, 1, device=DEVICE))

                # KS loss on a large fresh batch: the empirical KS deviation
                # at 32k points fluctuates ~0.004 -- noise floor swamps the
                # ~0.002 signal. 2048 samples = 1M points/channel puts the
                # sampling floor near 0.0006, below the ~0.0015 deviation
                # the 2M-point test can see, so the gradient is real.
                z_big = torch.randn(2048, Z_DIM, device=DEVICE)
                b_big = (torch.rand(2048, MSG_LEN, device=DEVICE) > 0.5).float() * 2 - 1
                fake_big = G(z_big, b_big)
                loss_g = (W_REC * g_rec + ramp * W_DET * g_det +
                          ramp * (W_MEAN * loss_mean(fake) + W_VAR * loss_var(fake) +
                          W_KURT * loss_kurtosis(fake) + W_SPEC * loss_spectral(fake, real) +
                          W_CYCLO * loss_cyclo(fake) + W_PSDCV * loss_psd_cv(fake) + 10.0 * loss_iq_match(fake) + W_XCORR * loss_iq_xcorr(fake) +
                          W_DIV * loss_diversity(fake) + W_QUANT * loss_quantile(fake) + 6.0 * loss_ks(fake_big)))

                g_opt.zero_grad(); dec_opt.zero_grad()
                loss_g.backward()
                # ep 622 collapsed (adv 93%) after a loss spike -- clip the
                # joint gradient so one bad batch cannot kick G into a
                # degenerate mode
                torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(Dec.parameters(), 5.0)
                g_opt.step(); dec_opt.step()
                g_losses.append(loss_g.item()); rec_losses.append(g_rec.item())

                # extra decoder-only steps (detached fake): accelerates BER
                for _ in range(4):
                    z2 = torch.randn(B, Z_DIM, device=DEVICE)
                    with torch.no_grad():
                        fake2 = G(z2, b)
                    dloss = F.binary_cross_entropy(Dec(fake2), bits)
                    dec_opt.zero_grad(); dloss.backward(); dec_opt.step()

        ber = quick_ber(G, Dec)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"G:{np.mean(g_losses):.3f} BCE:{np.mean(rec_losses):.4f} | "
              f"monitor-BER:{ber*100:.2f}%", flush=True)

        torch.save({'epoch': epoch, 'generator': G.state_dict(), 'decoder': Dec.state_dict(),
                    'g_opt': g_opt.state_dict()}, args.out)
        det = train_detector(G, dataset.gold_bank, steps=60)  # refresh for next epoch
        print(f"  detector acc on fresh gen: {detector_accuracy(det, G)*100:.1f}%", flush=True)

    print("\nTraining complete. Run: python test_metrics.py && python export_for_grc.py")

if __name__ == '__main__':
    main()
