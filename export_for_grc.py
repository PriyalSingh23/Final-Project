#!/usr/bin/env python3
"""Export trained models to TorchScript for GNU Radio"""
import torch
from models import Generator, Decoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'lpi_checkpoint.pt'

G = Generator(msg_len=256, emb_dim=32, z_dim=64, out_len=1024).to(DEVICE).eval()
D = Decoder(msg_len=256).to(DEVICE).eval()
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
G.load_state_dict(ckpt['generator'])
D.load_state_dict(ckpt['decoder'])

z_ex = torch.randn(1, 64, device=DEVICE)
b_ex = torch.randn(1, 256, device=DEVICE)
x_ex = torch.randn(1, 2, 512, device=DEVICE)

torch.jit.trace(G, (z_ex, b_ex)).save('generator_lpi.pt')
torch.jit.trace(D, x_ex).save('decoder_lpi.pt')
print("Exported: generator_lpi.pt, decoder_lpi.pt")
