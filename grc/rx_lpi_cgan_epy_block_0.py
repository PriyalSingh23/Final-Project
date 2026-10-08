import os, numpy as np
from gnuradio import gr

class cgan_decoder_block(gr.sync_block):
    def __init__(self, model_path='decoder_lpi.pt'):
        gr.sync_block.__init__(self, name='CGAN_LPI_Decoder', in_sig=[(np.complex64, 512)], out_sig=[(np.complex64, 256)])
        import torch
        self.torch = torch
        self.device = 'cpu'
        if not os.path.exists(model_path):
            for cand in ['decoder_lpi.pt', 'C:/Users/gspra/OneDrive/Desktop/LPI_CGAN/decoder_lpi.pt', '../decoder_lpi.pt']:
                if os.path.exists(cand):
                    model_path = cand
                    break
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.gpu_input = torch.zeros(1, 2, 512, dtype=torch.float32, device=self.device)
        print(f'[CGAN Dec] Loaded from {model_path} on {self.device}')

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        for i in range(len(in0)):
            frame = in0[i]
            v = np.concatenate([frame.real, frame.imag]).astype(np.float32)
            v[:512] -= v[:512].mean()
            v[512:] -= v[512:].mean()
            v = v / np.sqrt((v * v).mean() + 1e-8)
            self.gpu_input[0, 0, :] = self.torch.from_numpy(v[:512])
            self.gpu_input[0, 1, :] = self.torch.from_numpy(v[512:])
            with self.torch.no_grad():
                b_hat = self.model(self.gpu_input).cpu().numpy()[0]
            symbols = np.where(b_hat > 0.5, 1.0, -1.0).astype(np.complex64)
            out[i] = symbols
        return len(in0)
