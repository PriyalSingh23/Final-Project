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


class cgan_decoder_block(gr.sync_block):
    """CGAN Covert LPI Signal Decoder.
    Accepts vectors of 512 complex baseband samples, performs standard normalization,
    and infers 256 decoded BPSK symbols (-1.0 / +1.0) using the trained neural receiver.
    """

    def __init__(self, model_path="decoder_lpi.pt"):
        gr.sync_block.__init__(
            self,
            name="CGAN_LPI_Decoder",
            in_sig=[(np.complex64, 512)],
            out_sig=[(np.complex64, 256)],
        )
        self.model_path = str(model_path)
        self.torch = None
        self.model = None
        self.device = "cpu"
        self.gpu_input = None
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
                os.path.join(r"C:\Users\gspra\OneDrive\Desktop\LPI_CGAN", "decoder_lpi.pt"),
                os.path.join(os.path.dirname(__file__), "decoder_lpi.pt") if "__file__" in globals() else "",
                os.path.join(os.path.dirname(__file__), "..", "decoder_lpi.pt") if "__file__" in globals() else "",
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    resolved_path = os.path.abspath(cand)
                    break

            if resolved_path and os.path.exists(resolved_path):
                self.model = torch.jit.load(resolved_path, map_location=self.device)
                self.model.eval()
                self.gpu_input = torch.zeros(1, 2, 512, dtype=torch.float32, device=self.device)
                print(f"[CGAN Dec] Successfully loaded TorchScript model from {resolved_path}")
            else:
                print(f"[CGAN Dec] Note: Model '{raw_path}' will be loaded when flowgraph starts.")
        except Exception as e:
            print(f"[CGAN Dec] Note during inspection: {e}")

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
                out[i] = np.sign(in0[i][:256].real).astype(np.complex64)
            return num_frames

        for i in range(num_frames):
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

        return num_frames
