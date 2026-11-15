"""Detector: audio -> per-window fake probability, verdict and a suspicion heatmap."""
import numpy as np
import torch

from . import models
from .data import SR, window_starts, windows


def risk(p, threshold):
    """Display scale shared with the phone apps: the decision threshold maps to 0.5, so
    "risk >= 50%" always means the FAKE verdict, whatever the model's threshold is."""
    p = np.asarray(p, dtype=np.float64)
    return np.where(p < threshold, 0.5 * p / threshold, 0.5 + 0.5 * (p - threshold) / (1 - threshold))


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["model"] == "w2v":  # shape depends on the backbone used for caching
        state = ck["state"]
        net = models.W2VHead(n_layers=state["layer_w"].numel(), dim=state["norm.weight"].numel())
    else:
        net = models.build(ck["model"], ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


class Detector:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, ck = load_checkpoint(ckpt_path, self.device)
        self.kind, self.cfg = ck["model"], ck["cfg"]
        self.threshold = float(ck.get("threshold", 0.5))  # probability at dev-set EER
        self.n = int(self.cfg["audio"]["seconds"] * SR)
        self.hop = int(self.cfg["audio"]["hop_seconds"] * SR)
        self.backbone = None
        if self.kind == "w2v":
            self.backbone = models.XLSRFeatures(self.cfg["w2v"]["model_name"]).to(self.device)
            if self.device == "cuda":
                self.backbone.half()

    @torch.no_grad()
    def probs(self, wins, batch=16):
        """(W, N) float32 windows -> (W,) fake probabilities."""
        out = []
        for i in range(0, len(wins), batch):
            x = torch.from_numpy(np.ascontiguousarray(wins[i:i + batch])).to(self.device)
            if self.backbone is not None:
                x = self.backbone(x.half() if self.device == "cuda" else x)
            out.append(torch.sigmoid(self.net(x).float()).cpu().numpy())
        return np.concatenate(out)

    def analyse(self, x, max_windows=30):
        wins = windows(x, self.n, self.hop, max_windows)
        p = self.probs(wins)
        overall = float(p.mean())
        return {
            "window_probs": p,
            "window_starts": np.array(window_starts(len(x), self.n, self.hop, max_windows)) / SR,
            "fake_prob": overall,
            "risk": float(risk(overall, self.threshold)),
            "label": "FAKE" if overall >= self.threshold else "REAL",
            "threshold": self.threshold,
        }

    def heatmap(self, x, t_bins=16, f_bins=8):
        """Occlusion sensitivity on the first window: blank one time x frequency patch of the
        STFT, re-score, and record how much the fake probability drops. High = suspicious."""
        n_fft, hop = 512, 128
        w = windows(x, self.n, self.hop, 1)[0]
        win = torch.hann_window(n_fft)
        spec = torch.stft(torch.from_numpy(w), n_fft, hop, window=win, return_complex=True)
        F_, T_ = spec.shape
        te = np.linspace(0, T_, t_bins + 1).astype(int)
        fe = np.linspace(0, F_, f_bins + 1).astype(int)
        variants = []
        for fi in range(f_bins):
            for ti in range(t_bins):
                s = spec.clone()
                s[fe[fi]:fe[fi + 1], te[ti]:te[ti + 1]] = 0
                variants.append(torch.istft(s, n_fft, hop, window=win, length=len(w)).numpy())
        base = self.probs(w[None])[0]
        p = self.probs(np.stack(variants).astype(np.float32))
        heat = np.clip(base - p, 0, None).reshape(f_bins, t_bins)
        return heat / (heat.max() + 1e-9), w
