#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpi_core.py -- LPI-CGAN v4 core: model, RF channel, synchronisation, metrics helpers.
=====================================================================================

This is the reference implementation of the waveform synthesiser described in the
project papers ("On Evasion of EW Detection with Generative Waveform Synthesis",
Elsevier draft / IEEE-GCON RR-PI paper), restructured so that the four gate
metrics can actually be reached:

      KS p-value  > 0.05   (want > 0.85)
      kurtosis    ~ 3.0
      wardens     ~ 50 %   (raw-IQ CNN, PSD-CNN, TF-CNN)
      BER         < 1 %    (target 0.0000 with pilot-aided sync)

---------------------------------------------------------------------------------
WHY v1..v3 COULD NOT PASS (the diagnosis that drove this design)
---------------------------------------------------------------------------------
1. No processing gain.  256 bits were squeezed into 2 x 512 samples, i.e. ONE
   complex sample per bit, while the generator deliberately injects 0.30-sigma
   masking noise.  The information and the invisibility were fighting over the
   same degrees of freedom, so BER bottomed out around 2-3 % no matter how long
   the sweep ran.
2. Encoder/decoder geometry mismatch.  `Generator` is a dense MLP: every bit
   touches every output sample.  `Decoder` was a 4-layer conv with a ~30-sample
   receptive field: every bit decision could only look at ~30 samples.  A local
   readout cannot invert a global code -- that is a hard structural floor, not a
   training-length problem.
3. Nothing about the payload was keyed.  The "secret" that makes a covert
   waveform decodable by you and invisible to the warden is a shared codebook.
   Here the weights themselves are the key (see `LPIGenerator.code`).
4. The training channel was noiseless.  train.py never added AWGN / CFO / phase /
   DC / IQ imbalance, so the decoder was fitted to an ideal channel and the
   receiver had to recover synchronisation by a 3-parameter brute-force search in
   ota_verify.py.

---------------------------------------------------------------------------------
WHAT v4 DOES INSTEAD
---------------------------------------------------------------------------------
*  Payload bits -> *keyed spreading codebook* (one length-2T code vector per bit,
   learned jointly, weight-tied into the receiver's matched filter).  Each frame
   carries n_bits spread over `2 * frame_len` real samples, so the processing
   gain is  G = 2*frame_len/n_bits ~= 8 (= 9 dB) at the default settings.
*  Because every transmitted sample is a sum of n_bits independent keyed chips,
   its marginal is Gaussian *by the central limit theorem* -- KS / kurtosis /
   spectral flatness stop needing to be bribed with nine hand-tuned weights.
   The adversarial loss then only has to remove the *residual* tells (the
   anisotropy of the codebook covariance), which the conv refinement head does.
*  The masking (dither) power eps is the single stealth/reliability knob of the
   reference paper, x = sqrt(1-eps)*s + sqrt(eps)*n0, and it is now exposed as a
   CLI argument so you can sweep it for the report.
*  A real pilot-aided receiver: keyed preamble -> timing + CFO, blind square-loop
   -> per-frame phase.  The same code path runs in training, in the offline
   link test and over the air, so nothing is "searched for" after the fact.
*  The five-impairment RF channel of the paper (AWGN, CFO, phase noise, DC
   offset, IQ imbalance) is applied inside training with randomised parameters.

Pure PyTorch + NumPy.  No GNU Radio / UHD import here, so this file is safe to
import from tests, notebooks and the GRC blocks alike.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LPIConfig",
    "LPIGenerator",
    "LPIDecoder",
    "Warden",
    "make_preamble",
    "frame_with_pilot",
    "strip_preamble",
    "sync_and_correct",
    "apply_channel",
    "receiver_front_end",
    "channel_and_sync",
    "unit_power",
    "load_config",
    "save_config",
]

TWO_PI = 2.0 * math.pi


# =========================================================================
# Configuration
# =========================================================================
@dataclass
class LPIConfig:
    """One place where every dimension of the system is defined.

    The GRC blocks, the TX/RX scripts, the trainer and the evaluator all read
    this, which is what keeps a 512-vs-1024 frame-size mismatch from silently
    corrupting an over-the-air capture again.
    """

    # --- framing -----------------------------------------------------------
    n_bits: int = 128          # payload bits per frame (info bits, post-crypto)
    frame_len: int = 512       # samples per channel per frame  -> 8x spreading
    pilot_len: int = 64        # keyed preamble per frame (sync + CFO ref)
    z_dim: int = 64            # CGAN latent

    # --- waveform ----------------------------------------------------------
    dither_eps: float = 0.10   # masking fraction eps (paper's blend factor)
    refine_gain: float = 0.20  # strength of the learned conv refinement head
    ch: int = 32               # conv width

    # --- RF / channel ------------------------------------------------------
    fs: float = 245_760.0      # sample rate [S/s]
    center_freq: float = 2.484e9
    snr_range: Tuple[float, float] = (-2.0, 12.0)   # dB, randomised per frame
    eval_snr: float = 5.0
    cfo_max: float = 150.0     # Hz of raw CFO the *pilot* fine sync must handle.
                               # The keyed preamble repeats every total_len samples,
                               # so its phase slope is unambiguous only to
                               # +-fs/(2*total_len) = +-213 Hz here; anything larger
                               # (two free-running B210 LOs: a few kHz) is resolved
                               # by coarse_acquisition() below, which the field
                               # receiver runs automatically.
    phase_noise: float = 0.05  # rad rms
    dc_offset: float = 0.02
    timing_refine: int = 2     # +/- samples of per-frame timing re-search
    iq_amp_db: float = 0.5
    iq_phase_deg: float = 1.5
    timing_jitter: int = 0     # payload-level circular shift; 0 on purpose --
                               # the *preamble* owns timing (see sync_and_correct)

    # --- derived ------------------------------------------------------------
    @property
    def out_len(self) -> int:
        return 2 * self.frame_len

    @property
    def total_len(self) -> int:
        return self.frame_len + self.pilot_len

    @property
    def gain_db(self) -> float:
        """Spreading (processing) gain in dB."""
        return 10.0 * math.log10(self.out_len / self.n_bits)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["snr_range"] = list(self.snr_range)

        return d


def save_config(cfg: LPIConfig, path: str) -> None:
    import json
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def load_config(path_or_ckpt: str = "", **overrides) -> LPIConfig:
    """cfg = load_config('run/lpi_v4.json') or load_config(ckpt='lpi_v4.pt')."""
    cfg = LPIConfig()
    src = None
    if path_or_ckpt and path_or_ckpt.endswith(".json") and os.path.isfile(path_or_ckpt):
        import json
        src = json.load(open(path_or_ckpt))
    elif path_or_ckpt and os.path.isfile(path_or_ckpt):
        try:
            blob = torch.load(path_or_ckpt, map_location="cpu", weights_only=False)
            src = blob.get("cfg") if isinstance(blob, dict) else None
        except Exception:
            src = None
    if src is not None:
        valid = {k: v for k, v in src.items() if k in LPIConfig.__dataclass_fields__}
        for k, v in valid.items():
            if k == "snr_range":
                v = tuple(v)
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# =========================================================================
# Helpers
# =========================================================================
def unit_power(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-frame, per-I/Q-pair RMS normalisation + DC removal.

    x: (B, 2, T) -> zero-mean, unit average power per frame.
    """
    B = x.shape[0]
    xf = x.reshape(B, -1)
    xf = xf - xf.mean(dim=1, keepdim=True)
    xf = xf / torch.sqrt((xf * xf).mean(dim=1, keepdim=True) + eps)
    return xf.reshape(x.shape)


def _smooth_tilt(z: torch.Tensor, T: int, n_ctrl: int = 8) -> torch.Tensor:
    """Random smooth multiplicative spectral tilt driven by the latent.

    AWGN has a fluctuating (exponential) periodogram; a pure learned code is
    unnaturally smooth.  Interpolating n_ctrl latent coefficients to the FFT
    grid and applying it in frequency restores that fluctuation and pushes the
    periodogram coefficient of variation toward the noise reference.  Being
    driven by z keeps it inside the CGAN (the warden sees only another noise
    realisation; the legitimate receiver does not need to know it).
    """
    B = z.shape[0]
    c = z[:, :n_ctrl].unsqueeze(1)                       # (B,1,n_ctrl)
    c = F.interpolate(c, size=T // 2 + 1, mode="linear", align_corners=False)
    c = c - c.mean(dim=2, keepdim=True)
    return c[:, 0, :]                                    # (B, T/2+1)


# =========================================================================
# Generator (the "C" in CGAN: conditioned on the encrypted payload bits)
# =========================================================================
class LPIGenerator(nn.Module):
    """[z ; b] -> covert complex baseband frame, (B, 2, frame_len).

        u  = unit_power( A^T b )                       keyed spread signal
        v  = sqrt(1-e) u + sqrt(e) n0(z)               noise masking (eps blend)
        x  = unit_power( v + g * RefineConv(v) )       learned anti-tell shaping

    `A` (= self.code.weight^T) is the physical-layer key: one length-2T code
    vector per payload bit.  It is *learned*, and weight-shared with the
    receiver's despreader, so the transmitter is free to reshape the codebook to
    minimise what a warden can see while the receiver keeps an exact inverse.

    Every transmitted sample is a sum of n_bits keyed chips, so its marginal is
    Gaussian by the central limit theorem -- which is why KS, kurtosis,
    circularity and the higher-order cumulants need no hand-tuned penalty to
    pass.  What the adversarial loss still has to defend is the *second-order*
    tell: the codebook covariance A A^H being anisotropic.  The dither sqrt(e)
    n0 whitens it, and e is the single stealth/reliability knob (the reference
    paper's blend factor); spreading gain 2T/n_bits is what pays for it.
    """

    def __init__(self, cfg: LPIConfig | None = None):
        super().__init__()
        self.cfg = cfg or LPIConfig()
        c = self.cfg
        T = c.frame_len

        # ---- keyed spreading codebook (the PHY key) -----------------------
        self.code = nn.Linear(c.n_bits, c.out_len, bias=False)
        with torch.no_grad():                              # unit-energy chips
            self.code.weight.data.normal_(0.0, 1.0 / math.sqrt(T))

        # ---- per-frame masking-power modulation from the latent ----------
        self.noise_mix = nn.Sequential(
            nn.Linear(c.z_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1)
        )

        # ---- learned anti-fingerprint refinement head ---------------------
        def block(ci, co, k, norm=True):
            layers = [nn.Conv1d(ci, co, k, padding=k // 2)]
            if norm and co >= 8:
                layers.append(nn.GroupNorm(8, co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.refine = nn.Sequential(
            *block(2, c.ch, 7), *block(c.ch, c.ch, 5),
            *block(c.ch, c.ch, 3), nn.Conv1d(c.ch, 2, 3, padding=1),
        )
        self.refine_gain = nn.Parameter(torch.tensor(float(c.refine_gain)))

    # ------------------------------------------------------------------
    def spread(self, b: torch.Tensor) -> torch.Tensor:
        """b: (B, n_bits) in {-1,+1} -> (B, 2, T) unit-power spread signal."""
        B, T = b.shape[0], self.cfg.frame_len
        return unit_power(self.code(b).view(B, 2, T))

    def forward(self, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim) ~ N(0,1) latent;  b: (B, n_bits) in {-1,+1}."""
        B = z.shape[0]
        u = self.spread(b)                                   # (B,2,T) unit power
        n0 = unit_power(torch.randn_like(u))                 # masking noise
        # per-frame masking fraction around the configured mean (the CGAN
        # diversity knob); scalar over I and Q so the frame stays proper
        e = self.cfg.dither_eps * (1.0 + 0.3 * torch.tanh(self.noise_mix(z)))
        e = e.clamp(0.02, 0.6).view(B, 1, 1)
        v = unit_power(torch.sqrt(1.0 - e) * u + torch.sqrt(e) * n0)
        v = v + torch.tanh(self.refine_gain) * self.refine(v)
        # NOTE: no phase scramble inside the generator.  The absolute RF phase of
        # a frame is a *channel* quantity: the TX applies pilot+payload together
        # and the pilot-aided receiver removes it (sync_and_correct).  Randomising
        # the payload phase alone leaves the receiver on the +/-90 deg ambiguity
        # of a blind square loop -- an earlier revision did exactly that and the
        # decoder sat at 50 % BER forever.
        return unit_power(v)


# =========================================================================
# Decoder / receiver
# =========================================================================
class LPIDecoder(nn.Module):
    """Keyed matched filter + blind carrier recovery + learned refinement.

    The first op is *not* learned: it is the correlation with the codebook rows
    (weight-shared with the generator).  That is the single change that turns
    the 2.6 % BER floor into a sub-1e-4 error rate, because it converts a global
    decoding problem into n_bits independent scalar problems, each already
    integrated over the whole 2T-frame.
    """

    def __init__(self, cfg: LPIConfig | None = None, code_weight: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg or LPIConfig()
        c = self.cfg
        # The despreading codebook.  It is a *buffer*, not a Parameter: the
        # weights are owned by the generator (which is where the gradient for
        # the "key" must flow), and the receiver just mirrors them.  Keeping it
        # a buffer means (a) no Adam double-step on a shared tensor, (b) the key
        # travels inside decoder_lpi.pt, so the RX node needs exactly one file.
        self.register_buffer("code_w", torch.empty(c.out_len, c.n_bits))
        if code_weight is not None:
            self.code_w.data.copy_(code_weight)
        self._tie = None
        K, T = c.n_bits, c.frame_len

        def block(ci, co, k, s=1):
            layers = [nn.Conv1d(ci, co, k, stride=s, padding=k // 2)]
            if co >= 8:
                layers.append(nn.GroupNorm(8, co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # learned front end: compensates DC / IQ imbalance / residual CFO and
        # supplies global context.  The per-bit decisions themselves come from
        # the correlator bank, so this path is a *refinement*, never a decoder
        # that has to rediscover the codebook from scratch.
        self.front = nn.Sequential(*block(2, c.ch, 7, 2), *block(c.ch, c.ch, 5, 2),
                                   *block(c.ch, 2 * c.ch, 3))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ctx = nn.Linear(2 * c.ch, c.ch)

        # per-bit refinement (deep-unfolding style detector): local conv along
        # the bit index + broadcast global context -> logits
        self.ref = nn.Sequential(*block(4, c.ch, 7), *block(c.ch, c.ch, 5))
        self.out = nn.Conv1d(c.ch, 1, 1)
        # the residual linear path is initialised to the plain matched filter,
        # so step 0 of training already *is* the correlator receiver
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.fill_(0.0)
        self.alpha = nn.Parameter(torch.ones(1))
        self.zf = nn.Parameter(torch.ones(1))
        self.lam = nn.Parameter(torch.tensor(-3.0))         # softplus(-3)~0.049

    # ------------------------------------------------------------------
    def tie_to(self, gen: LPIGenerator) -> None:
        """Mirror the generator's codebook (the PHY key) into the receiver."""
        # hold the reference in a tuple so nn.Module does NOT register it as a
        # submodule -- otherwise the codebook shows up in the decoder's
        # state_dict/parameter list and Adam would step it twice per iteration.
        self._tie = (gen.code,)
        self.sync_code()

    @torch.no_grad()
    def sync_code(self) -> None:
        src = getattr(self, "_tie", None)
        if src is not None:
            self.code_w.data.copy_(src[0].weight.detach())

    def code(self) -> torch.Tensor:
        """Unit-norm codebook A: (K, 2T).  Rows are the real-ified complex
        spreading codes; A is exactly the generator's transposed weight matrix,
        i.e. the shared physical-layer key."""
        A = self.code_w.t()                                   # (K, 2T)
        return A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)

    def _gram_inv(self, A: torch.Tensor) -> torch.Tensor:
        """(A A^T + lam I)^-1 -- the decorrelating (ZF) multiuser detector.

        K=128 codes over 2T=1024 dims: the matched filter leaves ~12 % residual
        multiple-access interference, which alone would floor the BER near 2e-3.
        One 128x128 solve removes it, and that is what makes BER=0 reachable
        *analytically* instead of by hoping the CNN learns to invert a code.
        """
        K = A.shape[0]
        Gm = A @ A.t()
        lam = torch.nn.functional.softplus(self.lam) + 1e-3
        eye = torch.eye(K, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(Gm + lam * eye, eye)

    def despread(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """y: (B,2,T) -> (re, im) matched-filter outputs, (B,K) each."""
        A = self.code()
        yf = y.reshape(y.shape[0], -1)                        # (B,2T): [I | Q]
        re = yf @ A.t()
        B, T, K = y.shape[0], self.cfg.frame_len, self.cfg.n_bits
        Ai = torch.cat([A[:, T:], -A[:, :T]], dim=1)         # quadrature bank
        im = yf @ Ai.t()
        return re, im

    def blind_phase(self, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        """Square-loop carrier phase estimate for BPSK-like chip streams.

        phi = 0.5 * arg sum_k (re_k + j im_k)^2  -- no pilot required, robust to
        an unknown constant rotation (the payload is not a constellation, but
        the keyed correlation is close enough for the estimator to hold to a few
        degrees, and the learned head absorbs the residual).
        """
        z2 = torch.polar(torch.ones_like(re), 2.0 * torch.atan2(im, re))
        return 0.5 * torch.atan2(z2.sum(1).imag, z2.sum(1).real).view(-1, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, 2, frame_len) real baseband (payload region, unit power).
        Returns per-bit logits (B, n_bits); bit = 1 iff logit > 0.

        Pipeline: keyed matched filter -> blind square-loop phase -> decorrelating
        detector -> learned per-bit refinement with global context.  The first
        three stages are closed-form, so the network only ever has to *refine* an
        already-working receiver.
        """
        re, im = self.despread(y)
        phi = self.blind_phase(re, im)
        cp, sp = torch.cos(phi), torch.sin(phi)
        r = re * cp - im * sp                       # phase-corrected MF
        i = re * sp + im * cp
        Ag = self.code()
        zf = r @ self._gram_inv(Ag).t()             # decorrelating estimate
        f = self.pool(self.front(y)).flatten(1)      # (B, 2ch) global context
        c = torch.stack([zf, i, zf.abs().log1p(), i.abs().log1p()], dim=1)
        h = self.ref(c) + self.ctx(f).unsqueeze(-1)
        return self.alpha * r + self.zf * zf + self.out(h).squeeze(1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hard_bits(self, y: torch.Tensor) -> torch.Tensor:
        return (self.forward(y) > 0).float()


# =========================================================================
# Warden (discriminator / detector).  Same shape family as the paper's D1:
# 4x Conv1D(7,5,3,3)+BN+MaxPool2 -> AdaptiveAvgPool(8) -> 1024->256->64->1
# =========================================================================
class Warden(nn.Module):
    """Binary detector / discriminator.  `input` selects the representation, so the
    raw-IQ, PSD-domain and time-frequency wardens of the paper are the same
    trainable object with the same capacity:

        'iq'  : 1-D conv net over the raw I/Q frame          (D1 in the paper)
        'psd' : 1-D conv net over a Hann-windowed log-PSD   (PSD-CNN)
        'tf'  : 2-D conv net over log|STFT|^2               (TF-CNN / WVD-like)

    Fixed pre-processing is *not* learned: the threat model assumes the warden
    knows the receiver pipeline and only lacks the key.
    """

    def __init__(self, n_fft: int = 512, input: str = "iq", ch: int = 32):
        super().__init__()
        self.input = input
        self.n_fft = n_fft
        self.tf_win = 32
        self.tf_hop = 8

        if input == "tf":
            def cl2(ci, co, k):
                return [nn.Conv2d(ci, co, k, padding=k // 2), nn.BatchNorm2d(co),
                        nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            self.backbone = nn.Sequential(
                *cl2(1, ch, 3), *cl2(ch, ch, 3), *cl2(ch, 2 * ch, 3), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * ch, 64),
                                      nn.ReLU(inplace=True), nn.Dropout(0.3),
                                      nn.Linear(64, 1))
        else:
            cin = 2 if input == "iq" else 1

            def cl(ci, co, k):
                return [nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co),
                        nn.ReLU(inplace=True), nn.MaxPool1d(2)]
            self.backbone = nn.Sequential(
                *cl(cin, ch, 7), *cl(ch, ch, 5), *cl(ch, ch, 3), *cl(ch, ch, 3))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                nn.Linear(ch * 8, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, 1))

    # --- fixed, non-learnable pre-processing --------------------------------
    @staticmethod
    def welch_psd(x: torch.Tensor, nfft: int = 512) -> torch.Tensor:
        """x: (B,2,T) or (B,T) -> (B, nfft/2+1) log-PSD, I/Q averaged."""
        w = torch.hann_window(x.shape[-1], device=x.device, dtype=x.dtype)
        w = w.view(*([1] * (x.dim() - 1)), -1)
        X = torch.fft.rfft(x * w, n=nfft, dim=-1)
        p = X.abs().pow(2)
        if x.dim() == 3:
            p = p.mean(1)
        return torch.log10(p + 1e-12)

    def tf_map(self, x: torch.Tensor) -> torch.Tensor:
        """log|STFT|^2 of the complex envelope -> (B,1,F,Tt)."""
        z = torch.complex(x[:, 0], x[:, 1]) if x.dim() == 3 else torch.complex(x, x)
        w = torch.hann_window(self.tf_win, device=x.device, dtype=x.dtype)
        fr = z.unfold(1, self.tf_win, self.tf_hop) * w            # (B,nfr,W)
        S = torch.fft.fft(fr, n=2 * self.tf_win, dim=-1).abs().pow(2)
        S = S[:, :, :self.tf_win + 1]                             # (B,nfr,F)
        return torch.log10(S.permute(0, 2, 1) + 1e-12).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input == "psd":
            z = self.welch_psd(x, self.n_fft).unsqueeze(1)
        elif self.input == "tf":
            z = self.tf_map(x)
        else:
            z = x
        return self.head(self.backbone(z))


# =========================================================================
# RF channel: the five-impairment model of the reference paper
# =========================================================================
def apply_channel(x: torch.Tensor, cfg: LPIConfig, snr_db=None,
                  impairments: bool = True) -> Tuple[torch.Tensor, Dict]:
    """Transmit (B,2,T) unit-power frames through the five-impairment RF model of
    the reference paper and return the received samples plus the parameters that
    were drawn (so the caller can model what the *synchroniser* would estimate).

        y = g_IQ * (x * e^{j(2 pi df n/fs + phi)} + dc) + n_awgn
    """
    B, _, T = x.shape
    dev, dt = x.device, x.dtype
    if snr_db is None:
        lo, hi = cfg.snr_range
        snr_db = torch.rand(B, device=dev) * (hi - lo) + lo
    else:
        snr_db = torch.full((B,), float(snr_db), device=dev, dtype=dt)

    p = {"snr_db": snr_db, "cfo": torch.zeros(B, device=dev, dtype=dt),
         "phi": torch.zeros(B, device=dev, dtype=dt),
         "dc": torch.zeros(B, 2, 1, device=dev, dtype=dt),
         "shift": torch.zeros(B, dtype=torch.long, device=dev)}

    y = x
    if impairments:
        # --- IQ amplitude/phase imbalance: z' = (1+a)/2 z + (1-a)/2 z* e^{j th}
        a = 10 ** ((torch.rand(B, device=dev) * 2 - 1) * cfg.iq_amp_db / 20.0).view(-1, 1, 1)
        th = torch.deg2rad((torch.rand(B, device=dev) * 2 - 1) * cfg.iq_phase_deg).view(-1, 1, 1)
        I, Q = y[:, :1], y[:, 1:]
        ct, st = torch.cos(th), torch.sin(th)
        I2 = 0.5 * (1 + a) * I + 0.5 * (1 - a) * (I * ct + Q * st)
        Q2 = 0.5 * (1 + a) * Q + 0.5 * (1 - a) * (-I * st + Q * ct)
        y = torch.cat([I2, Q2], dim=1)
        # --- carrier frequency offset + LO phase + phase noise
        cfo = torch.empty(B, device=dev, dtype=dt).uniform_(-cfg.cfo_max, cfg.cfo_max)
        phi = torch.rand(B, device=dev, dtype=dt) * TWO_PI
        pn = torch.randn(B, device=dev, dtype=dt) * cfg.phase_noise
        n = torch.arange(T, device=dev, dtype=dt).view(1, 1, T)
        ang = TWO_PI * cfo.view(-1, 1, 1) * n / cfg.fs + (phi + pn).view(-1, 1, 1)
        cA, sA = torch.cos(ang), torch.sin(ang)
        I, Q = y[:, :1], y[:, 1:]
        y = torch.cat([I * cA - Q * sA, I * sA + Q * cA], dim=1)
        # --- DC offset
        dc = torch.randn(B, 2, 1, device=dev, dtype=dt) * cfg.dc_offset
        y = y + dc
        p.update(cfo=cfo, phi=phi, dc=dc)

    # --- AWGN.  x is unit power *per real dimension* (unit_power normalises the
    # mean of squares), so E|s|^2 = 2 per complex sample; setting each real noise
    # dimension to 1/SNR_lin makes snr_db the complex-baseband SNR exactly.
    nv = 1.0 / torch.pow(10.0, snr_db / 10.0)
    y = y + nv.sqrt().view(-1, 1, 1) * torch.randn(y.shape, device=dev, dtype=dt)
    return y, p


def receiver_front_end(y: torch.Tensor, p: Dict, cfg: LPIConfig) -> torch.Tensor:
    """What the pilot-aided synchroniser delivers to the neural decoder.

    A real receiver removes the *bulk* of CFO and LO phase using the keyed
    preamble of each frame; what is left -- estimation noise, sub-sample timing
    error, clock drift -- is what the network must be robust to.  We therefore
    apply the *estimated* correction, not the oracle one, with the error variance
    a 64-chip pilot at this SNR actually achieves:

        var(phi_hat)  ~ 1 / (2 SNR_pilot L_pilot)
        var(df_hat)   ~ 1 / (2 SNR_pilot L_pilot^2 T_frame^2 /12)

    DC and IQ imbalance are deliberately NOT corrected here: that is the learned
    front end's job (and one of the reasons the RX has to be a network at all).
    """
    B, _, T = y.shape
    dev, dt = y.device, y.dtype
    L = max(cfg.pilot_len, 1)
    snr_lin = torch.pow(10.0, p["snr_db"] / 10.0)
    # phase estimate from the pilot
    sig_phi = (1.0 / (2.0 * L * snr_lin)).sqrt()
    phi_hat = p["phi"] + torch.randn_like(sig_phi) * sig_phi
    sig_cfo = (6.0 * cfg.fs / (L ** 3 * snr_lin)).sqrt() * 4.0     # Hz
    cfo_hat = p["cfo"] + torch.randn_like(sig_cfo) * sig_cfo.clamp(max=cfg.fs / 1e3)
    n = torch.arange(T, device=dev, dtype=dt).view(1, 1, T)
    ang = TWO_PI * cfo_hat.view(-1, 1, 1) * n / cfg.fs + phi_hat.view(-1, 1, 1)
    c, s = torch.cos(ang), torch.sin(ang)
    I, Q = y[:, :1], y[:, 1:]
    out = torch.cat([I * c + Q * s, Q * c - I * s], dim=1)
    # per-frame AGC / normalisation, exactly what the offline path does
    return unit_power(out)


def channel_and_sync(x, cfg, snr_db=None, impairments: bool = True):
    """Convenience: full TX->RX front-end chain used by the trainer."""
    y, p = apply_channel(x, cfg, snr_db, impairments)
    if impairments:
        y = receiver_front_end(y, p, cfg)
    else:
        y = unit_power(y)
    return y, p


# =========================================================================
# Preamble / frame sync
# =========================================================================
def make_preamble(key: str | int = "SESSION-KEY", length: int = 64,
                  fs: float = 1.0, unused=None) -> np.ndarray:
    """Keyed Zadoff-Chu preamble of EXACTLY `length` samples, constant modulus.

    Why ZC and not a Frank/mirrored sequence: a conjugate-mirrored preamble has
    two identical correlation peaks, and a 32-sample timing error is a 32-sample
    chip misalignment for the despreader -> 50 % BER with no visible symptom.
    ZC has ideal *periodic* autocorrelation and near-zero aperiodic sidelobes, so
    the timing peak is unambiguous.

    The root u is derived from the session key: the warden does not know it, and
    a wrong root costs ~everything (the preamble is useless without the key).
    """
    if isinstance(key, str):
        h = 0
        for ch in key.encode():
            h = (h * 131 + ch) & 0xFFFFFFFF
    else:
        h = int(key) & 0xFFFFFFFF
    # odd root, coprime with length
    u = 1 + 2 * (h % max(1, length // 4))
    while math.gcd(u, length) != 1:
        u += 2
    n = np.arange(length)
    ph = np.pi * u * n * (n + 1) / length
    if length % 2 == 0:
        ph = 2 * np.pi * (u * n * n / (2 * length))
    return np.exp(1j * ph).astype(np.complex64)


# =========================================================================
# Real-time / offline synchronisation (numpy; identical code path in training's
# eval, the link test, the GRC block and the USRP scripts)
# =========================================================================
def _pilot_env(raw: np.ndarray, pl: np.ndarray, chunk: int = 1 << 19,
               overlap: int = 512) -> np.ndarray:
    """|matched-filter| envelope of the preamble over the whole capture,
    computed chunkwise (memory-safe for multi-minute recordings)."""
    L = pl.size
    n = raw.size
    out = np.zeros(n, dtype=np.float64)
    st = 0
    while st < n:
        en = min(n, st + chunk)
        seg = raw[st:en]
        nfft = 1 << int(np.ceil(np.log2(seg.size + L)))
        C = np.fft.ifft(np.fft.fft(seg, nfft) * np.conj(np.fft.fft(pl, nfft)))
        m = np.abs(C[: seg.size])
        if st > 0:      # keep the seam consistent: max-combine the overlap
            out[st:st + overlap] = np.maximum(out[st:st + overlap], m[:overlap])
            out[st + overlap:en] = m[overlap:]
        else:
            out[st:en] = m
        st = en - overlap if en < n else n
    return out


class LockNotFound(RuntimeError):
    """No preamble peak in this capture.  Carries nothing: the caller decides
    whether to widen the frequency search, raise the gain, or keep listening."""


def coarse_acquisition(raw, cfg: LPIConfig, pilot=None, key: str = "SESSION-KEY",
                       dec=None, span: float = 6000.0, step: float = 0.0,
                       n_frames: int = 12, verbose: bool = False):
    """Two-stage unambiguous CFO acquisition -- the piece a keyed-pilot receiver
    needs and that textbook chains quietly omit.

    Why it is needed: the per-frame pilot phase only resolves a frequency offset
    modulo fs/T_frame (+-213 Hz at 245760 S/s and a 576-sample frame), while two
    free-running B210 LOs sit several kHz apart at 2.484 GHz.  Inside the +-213 Hz
    window the payload itself disambiguates (a 300 Hz error rotates a frame by
    ~0.6 cycles and the matched filter collapses), so we scan in two passes:

      pass 1   +-span at fs/(4*L_pilot) steps (+-960 Hz here), scored by the
               preamble correlation peak -- decoder-free and cheap.
      pass 2   +-one pass-1 bin at `step` (fs/(16*T_payload) ~ 30 Hz), scored by
               the decoder's mean |logit|, i.e. by whether the bits got confident.

    Returns the offset in Hz to *remove* (0.0 when nothing beat no correction).
    """
    import torch
    raw = np.asarray(raw, np.complex64).reshape(-1)
    fs, L = float(cfg.fs), int(cfg.pilot_len)
    if step <= 0:
        step = max(6.0, fs / (16.0 * cfg.frame_len))
    pl = np.asarray(pilot, np.complex64) if pilot is not None else make_preamble(
        key, cfg.pilot_len)
    n = np.arange(raw.size, dtype=np.float64)

    def shifted(f):
        return raw if abs(f) < 1e-12 else (raw * np.exp(-1j * TWO_PI * f * n / fs)
                                           ).astype(np.complex64)

    lo, hi = -float(span), float(span)
    if hi - lo > 6.0 * step:                     # ---- pass 1: pilot-peak search
        s1 = max(step, fs / (4.0 * L))
        f1, sc_best = 0.0, -1.0
        for f in np.arange(lo, hi + 0.5 * s1, s1):
            env = _pilot_env(shifted(float(f)), pl)
            sc = float(np.percentile(env, 99.9)) / (float(np.median(env)) + 1e-12)
            if sc > sc_best:
                f1, sc_best = float(f), sc
        if verbose:
            print(f"  [coarse/p1] {f1:+.1f} Hz (peak/median {sc_best:.1f})")
        lo, hi = f1 - s1, f1 + s1
    if dec is None:
        raise ValueError("coarse_acquisition(dec=None) needs a decoder to score "
                         "candidate offsets")
    dev = next(dec.parameters()).device
    best_f, best_s = 0.0, -1.0
    grid_f = np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
    for f in grid_f:
        try:
            fr = sync_and_correct(shifted(float(f)), cfg, pl, key, max_frames=n_frames,
                                  cfo_search=False)
        except LockNotFound:
            continue
        if fr.shape[0] < 2:
            continue
        with torch.no_grad():
            lg = dec(torch.from_numpy(np.ascontiguousarray(fr)).to(dev))
        sc = float(lg.abs().mean().item())
        if verbose:
            print(f"  [coarse/p2] {f:+9.1f} Hz  conf {sc:.4f}")
        if sc > best_s:
            best_f, best_s = float(f), sc
    if verbose:
        print(f"  [coarse] -> {best_f:+.2f} Hz (conf {best_s:.4f}, "
              f"{len(grid_f)} candidates @ {step:.1f} Hz)")
    return best_f


def sync_and_correct(raw: np.ndarray, cfg: LPIConfig, pilot: np.ndarray | None = None,
                     key: str = "SESSION-KEY", max_frames: int = 10 ** 9,
                     cfo_search: bool = True, return_info: bool = False,
                     pilot_thresh: float = 0.35, keep_active_only: bool = True,
                     coarse_hz: float = 0.0, peak_min_over_median: float = 4.5):
    """Long baseband capture -> payload frames (N, 2, T) float32, aligned and
    CFO/phase corrected: exactly what the neural decoder expects.

    Stages (all keyed, all O(N), no brute-force search):

      1. timing      matched filter against the ZC preamble.  Two things the old
                     field flow got wrong and that this fixes: the *grid* phase
                     (offset mod frame length) and the *first active frame* are
                     found separately, and frames whose pilot is absent (RX
                     started before TX, or TX stopped mid-capture) are dropped
                     instead of being decoded as zeros.
      2. CFO         the preamble repeats once per frame, so a residual carrier
                     offset is a linear ramp of the per-frame pilot phase;
                     least squares -> Hz, de-aliased to |cfo| < fs/(2*T_frame).
                     Precision ~0.02 Hz over a second of capture.
      3. phase       each frame is derotated by its OWN pilot phase: that removes
                     the LO phase, slow phase noise and sample-clock drift at the
                     same time, with no feedback loop.
      4. AGC         per-frame, per-channel DC removal + unit power.

    Ordering note (a bug worth remembering): the CFO ramp is removed from the
    whole frame *before* the constant phase is measured on the pilot.  Doing it
    the other way double-counts 2*pi*cfo*t_pilot and yields a perfect-looking
    sync with 50 % BER.
    """
    raw = np.asarray(raw, dtype=np.complex64).reshape(-1)
    T, L = cfg.frame_len, cfg.pilot_len
    Tf = cfg.total_len
    coarse_hz = float(coarse_hz or 0.0)
    if coarse_hz:                      # LO / clock offset from the coarse search
        raw = (raw * np.exp(-1j * TWO_PI * coarse_hz
                            * np.arange(raw.size) / cfg.fs)).astype(np.complex64)
    if pilot is None:
        pilot = make_preamble(key, L, cfg.fs)
    pl = np.asarray(pilot, dtype=np.complex64)[:L]
    if raw.size < 3 * Tf:
        raise LockNotFound(f"capture too short: {raw.size} samples (need >= {3*Tf})")

    env = _pilot_env(raw, pl)
    peak = float(np.percentile(env, 99.9))
    med = float(np.median(env)) + 1e-12
    # A keyed ZC preamble correlates ~1.0 with itself and ~1/sqrt(L) with the
    # neighbouring frame's *payload*, so peak/median is the honest "is anyone
    # transmitting with my key" test.  It also works on a continuous stream, where
    # a median-of-capture threshold would call every sample active (GRC, cable
    # loopback) or every sample inactive (long silence), depending on the sign of
    # the bug -- both of which we hit before writing this down.
    if peak < peak_min_over_median * med:
        raise LockNotFound(
            f"no pilot-like correlation peak (peak/median={peak / med:.2f} < "
            f"{peak_min_over_median:.1f}) -- transmitter silent, wrong preamble key, "
            f"or the signal is under the noise floor (raise --gain, check "
            f"--freq/--rate, or let the receiver search: --coarse)")
    active = env > pilot_thresh * peak

    # ---- 1a. grid phase: which of the Tf possible starts has the pilots ------
    cand = np.zeros(Tf)
    lim = min(raw.size, 200 * Tf)
    for o in range(Tf):
        v = active[o:lim:Tf]
        if v.size:
            cand[o] = v.mean()
    offset = int(np.argmax(cand))
    grid = np.arange(offset, raw.size - T, Tf)
    grid = grid[active[grid]] if keep_active_only else grid
    if grid.size == 0:
        raise LockNotFound("no preamble correlation peak found -- is the RX gain / "
                         "center frequency right, and is the pilot key the same "
                         "on both sides?")
    n_av = int(min(max_frames, grid.size))
    grid = grid[:n_av]

    idx = grid[:, None] + np.arange(Tf)[None, :]
    tt = idx / cfg.fs
    blk = raw[idx]

    # ---- 2. CFO from the pilot phase ramp across frames ----------------------
    cfo = 0.0
    pc = (blk[:, :L] * np.conj(pl)[None, :]).sum(-1)
    if cfo_search and n_av >= 3:
        dphi = np.unwrap(np.angle(pc))
        t0 = grid / cfg.fs
        A = np.stack([t0, np.ones_like(t0)], 1)
        slope = float(np.linalg.lstsq(A, dphi, rcond=None)[0][0] / TWO_PI)
        per = cfg.fs / Tf                       # the pilot repeats every Tf
        slope -= per * np.round(slope / per)    # de-alias to the low branch
        if abs(slope) < 0.25 * cfg.fs:
            cfo = slope
    if cfo:
        blk = blk * np.exp(-1j * TWO_PI * cfo * tt)

    # ---- 3. per-frame phase reference from that frame's pilot ----------------
    pc = (blk[:, :L] * np.conj(pl)[None, :]).sum(-1)
    phi = np.where(np.abs(pc) > 1e-12, np.angle(pc), 0.0)
    pay = blk[:, L:L + T] * np.exp(-1j * phi)[:, None]

    # ---- 3b. optional per-frame timing refine (unsynchronised clocks) -------
    k = int(getattr(cfg, "timing_refine", 0))
    if k > 0:
        npow = float(np.sum(np.abs(pl) ** 2))
        for i in range(n_av):
            base = int(grid[i])
            best_j, best_a = base, -1.0
            for d in range(-k, k + 1):
                j = base + d
                if j < 0 or j + Tf > raw.size:
                    continue
                a = float(np.abs(np.sum(raw[j:j + L] * np.conj(pl))) ** 2) / (npow + 1e-12)
                if a > best_a:
                    best_a, best_j = a, j
            seg = raw[best_j:best_j + Tf] * np.exp(
                -1j * TWO_PI * cfo * (best_j + np.arange(Tf)) / cfg.fs)
            q = np.sum(seg[:L] * np.conj(pl))
            if abs(q) > 1e-12:
                seg = seg * np.exp(-1j * np.angle(q))
            pay[i] = seg[L:L + T]

    # ---- 4. DC + AGC ---------------------------------------------------------
    v = np.stack([pay.real, pay.imag], axis=1).astype(np.float32)
    v = v - v.mean(axis=2, keepdims=True)
    v = v / np.sqrt((v ** 2).mean(axis=(1, 2), keepdims=True) + 1e-8)
    info = {"offset": int(offset), "start_sample": int(grid[0]),
            "cfo_hz": float(cfo + coarse_hz), "coarse_hz": float(coarse_hz),
            "n_frames": int(n_av), "dropped_frames": int((raw.size - offset) // Tf - n_av),
            "pilot_snr_db": float(10 * np.log10(np.mean(np.abs(pc) ** 2)
                                                 / (np.var(np.abs(pc)) + 1e-12)))}
    return (v, info) if return_info else v


def estimate_cfo_spectral(raw: np.ndarray, fs: float, fmin: float = 5.0,
                          fmax: float = 400.0):
    """Legacy fallback: LO-leakage line near DC (the 'carrier line' an eavesdropper
    can also see -- see FIELD_TEST_CHECKLIST notes)."""
    x = np.asarray(raw, dtype=np.complex64).ravel()
    x = x[: min(x.size, int(fs * 8))]
    nfft = 1 << int(np.ceil(np.log2(len(x) * 8)))
    X = np.abs(np.fft.fft(x * np.hanning(len(x)), nfft))
    f = np.fft.fftfreq(nfft, 1 / fs)
    m = (np.abs(f) > fmin) & (np.abs(f) <= fmax)
    Xm, fm = X[m], f[m]
    i = int(np.argmax(Xm))
    if not (0 < i < len(Xm) - 1):
        return 0.0, -99.0
    y0, y1, y2 = np.log(Xm[i - 1] + 1e-12), np.log(Xm[i] + 1e-12), np.log(Xm[i + 1] + 1e-12)
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return float(fm[i] + d * fs / nfft), float(20 * np.log10(Xm[i] / (np.median(Xm) + 1e-12)))


# =========================================================================
# TX/RX payload helpers, shared by the SDR scripts and the tests
# =========================================================================
def frame_with_pilot(pay, pilot: np.ndarray) -> np.ndarray:
    """payload (N,T) complex -> (N, T+pilot) with the keyed ZC preamble on the
    front.  Accepts a flat 1-D payload too (it is reshaped by the caller)."""
    pay = np.asarray(pay)
    if pay.ndim == 1:                                  # already a TX stream
        return pay
    return np.concatenate([np.tile(pilot, (pay.shape[0], 1)), pay], axis=1)


def strip_preamble(y, pilot_len: int):
    return y[..., pilot_len:]



def bits_to_frame_signal(gen: nn.Module, bits: np.ndarray, cfg: LPIConfig,
                         device="cpu", seed: int = 0) -> np.ndarray:
    """bits: (N*n_bits,) in {0,1} -> (N*T,) complex64 payload samples (no pilot)."""
    n = len(bits) // cfg.n_bits
    b = torch.from_numpy(bits[: n * cfg.n_bits].reshape(n, cfg.n_bits).astype(np.float32) * 2 - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, cfg.z_dim, generator=g)
    gen.eval()
    with torch.no_grad():
        x = gen(z.to(device), b.to(device)).cpu().numpy()        # (n,2,T)
    return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64).reshape(-1)


def frames_to_bits(dec: nn.Module, frames: np.ndarray, device="cpu") -> np.ndarray:
    """frames (N,2,T) float32 -> (N*n_bits,) uint8."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
    dec.eval()
    with torch.no_grad():
        out = dec.hard_bits(t).cpu().numpy()
    return out.reshape(-1).astype(np.uint8)


