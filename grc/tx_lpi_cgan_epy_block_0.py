import os, numpy as np
from gnuradio import gr

class cgan_generator_block(gr.sync_block):
    def __init__(self, model_path='generator_lpi.pt', gain_factor=0.35):
        gr.sync_block.__init__(self, name='CGAN_LPI_Generator', in_sig=[(np.complex64, 256)], out_sig=[(np.complex64, 512)])
        import torch
        self.torch = torch
        self.device = 'cpu'
        if not os.path.exists(model_path):
            for cand in ['generator_lpi.pt', 'C:/Users/gspra/OneDrive/Desktop/LPI_CGAN/generator_lpi.pt', '../generator_lpi.pt']:
                if os.path.exists(cand):
                    model_path = cand
                    break
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.gain = gain_factor
        self.gpu_noise = torch.zeros(1, 64, dtype=torch.float32, device=self.device)
        self.b_tensor = torch.zeros(1, 256, dtype=torch.float32, device=self.device)
        print(f'[CGAN Gen] Loaded from {model_path} on {self.device}')

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        for i in range(len(in0)):
            frame = in0[i]
            self.b_tensor[0, :] = self.torch.from_numpy(frame.real.astype(np.float32))
            self.gpu_noise.normal_(mean=0, std=1)
            with self.torch.no_grad():
                covert = self.model(self.gpu_noise, self.b_tensor)
            c_np = covert.cpu().numpy()[0]
            out[i] = (c_np[0, :] + 1j * c_np[1, :]).astype(np.complex64) * self.gain
        return len(in0)
