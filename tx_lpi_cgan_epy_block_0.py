import os
import sys
import numpy as np

# Ensure PyTorch and project modules are accessible across all conda environments
for _extra_path in [
    r"C:\Users\gspra\radioconda\Lib\site-packages",
    r"C:\Users\gspra\OneDrive\Desktop\LPI_CGAN",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in globals() else "",
]:
    if _extra_path and os.path.exists(_extra_path) and _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

from gnuradio import gr


class cgan_generator_block(gr.sync_block):
    """CGAN Covert LPI Signal Generator.
    Accepts vectors of 256 BPSK symbols and synthesizes 512 complex Gaussian covert baseband samples.
    """

    def __init__(self, model_path="generator_lpi.pt", gain_factor=0.35):
        gr.sync_block.__init__(
            self,
            name="CGAN_LPI_Generator",
            in_sig=[(np.complex64, 256)],
            out_sig=[(np.complex64, 512)],
        )
        self.model_path = str(model_path)
        self.gain_factor = float(gain_factor)
        self.torch = None
        self.model = None
        self.device = "cpu"
        self.gpu_noise = None
        self.b_tensor = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            self.torch = torch
            self.device = "cpu"
            raw_path = self.model_path.strip("'\"")
            resolved_path = None
            candidates = [
                raw_path,
                os.path.join(r"C:\Users\gspra\OneDrive\Desktop\LPI_CGAN", raw_path),
                os.path.join(r"C:\Users\gspra\OneDrive\Desktop\LPI_CGAN", "generator_lpi.pt"),
                os.path.join(os.path.dirname(__file__), "generator_lpi.pt") if "__file__" in globals() else "",
                os.path.join(os.path.dirname(__file__), "..", "generator_lpi.pt") if "__file__" in globals() else "",
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    resolved_path = os.path.abspath(cand)
                    break

            if resolved_path and os.path.exists(resolved_path):
                self.model = torch.jit.load(resolved_path, map_location=self.device)
                self.model.eval()
                self.gpu_noise = torch.zeros(1, 64, dtype=torch.float32, device=self.device)
                self.b_tensor = torch.zeros(1, 256, dtype=torch.float32, device=self.device)
                print(f"[CGAN Gen] Successfully loaded TorchScript model from {resolved_path}")
            else:
                print(f"[CGAN Gen] Note: Model '{raw_path}' will be loaded when flowgraph starts.")
        except Exception as e:
            print(f"[CGAN Gen] Note during inspection: {e}")

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        num_frames = len(in0)
        if num_frames == 0:
            return 0

        if self.model is None:
            self._load_model()

        if self.model is None or self.torch is None:
            # Fallback zero/pass-through if model weights unavailable
            for i in range(num_frames):
                out[i][:256] = in0[i]
                out[i][256:] = 0.0
            return num_frames

        for i in range(num_frames):
            frame = in0[i]
            self.b_tensor[0, :] = self.torch.from_numpy(frame.real.astype(np.float32))
            self.gpu_noise.normal_(mean=0.0, std=1.0)
            with self.torch.no_grad():
                covert = self.model(self.gpu_noise, self.b_tensor)
            c_np = covert.cpu().numpy()[0]
            out[i] = (c_np[0, :] + 1j * c_np[1, :]).astype(np.complex64) * self.gain_factor

        return num_frames
