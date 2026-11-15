"""The four models: GMM baseline, log-mel CNN, XLS-R head, tiny raw-audio student.

All neural models map input -> one logit (positive = fake).
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SR


# ---------------------------------------------------------------- GMM baseline

def mfcc_frames(x, n_mfcc=20):
    import librosa
    m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    return np.concatenate([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).T.astype(np.float64)


class GMMClassifier:
    """Two diagonal GMMs on MFCC+deltas; score = mean log-likelihood ratio (fake - real)."""

    def __init__(self, n_components=512, n_mfcc=20):
        from sklearn.mixture import GaussianMixture
        self.n_mfcc = n_mfcc
        self.gmms = [GaussianMixture(n_components, covariance_type="diag", max_iter=50,
                                     reg_covar=1e-3, random_state=0) for _ in range(2)]
        self.threshold = 0.0

    def fit(self, frames_real, frames_fake):
        self.gmms[0].fit(frames_real)
        self.gmms[1].fit(frames_fake)
        return self

    def score(self, x):
        f = mfcc_frames(x, self.n_mfcc)
        return float(self.gmms[1].score(f) - self.gmms[0].score(f))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


# ---------------------------------------------------------------- CNN on log-mel

class LogMel(nn.Module):
    def __init__(self, n_mels=80, n_fft=512, hop=160):
        super().__init__()
        import librosa
        fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=SR // 2)
        self.register_buffer("fb", torch.tensor(fb, dtype=torch.float32))
        self.register_buffer("win", torch.hann_window(n_fft))
        self.n_fft, self.hop = n_fft, hop

    def forward(self, x):  # (B, N) -> (B, 1, n_mels, T)
        s = torch.stft(x.float(), self.n_fft, self.hop, window=self.win, return_complex=True)
        m = torch.log(self.fb @ s.abs().pow(2) + 1e-6)
        m = m - m.mean(dim=(1, 2), keepdim=True)
        return m.unsqueeze(1)


class ResBlock2d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.skip(x))


class MelCNN(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.front = LogMel(n_mels)
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        chans = [16, 32, 64, 128]
        self.blocks = nn.Sequential(*[ResBlock2d(a, b, 2) for a, b in zip([16] + chans[:-1], chans)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def feature_map(self, x):
        return self.blocks(self.stem(self.front(x)))

    def forward(self, x):
        h = self.feature_map(x).mean(dim=(2, 3))
        return self.fc(self.drop(h)).squeeze(-1)


# ---------------------------------------------------------------- XLS-R (frozen) + head

class XLSRFeatures(nn.Module):
    """Frozen wav2vec 2.0 / XLS-R: waveform -> per-layer [mean, std] over time, (B, L, 2D)."""

    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)
        hs = torch.stack(self.model(x, output_hidden_states=True).hidden_states, 1)  # B,L,T,D
        return torch.cat([hs.mean(2), hs.std(2)], -1)


class W2VHead(nn.Module):
    """Learned softmax weighting over layers + MLP on pooled statistics."""

    def __init__(self, n_layers=25, dim=2048, hidden=256):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, feats):  # (B, L, 2D)
        w = torch.softmax(self.layer_w, 0)
        h = (feats.float() * w[None, :, None]).sum(1)
        return self.mlp(self.norm(h)).squeeze(-1)


# ---------------------------------------------------------------- tiny student (phone)

class DSConv(nn.Module):
    def __init__(self, cin, cout, k=9):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class StudentBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a, self.b = DSConv(cin, cout), DSConv(cout, cout)
        self.skip = nn.Conv1d(cin, cout, 1, bias=False)
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        y = self.b(F.relu(self.a(x)))
        return self.pool(F.relu(y + self.skip(x)))


class Student(nn.Module):
    """Raw 16 kHz waveform -> logit. ~0.2M params, only ONNX-friendly ops (no STFT)."""

    def __init__(self, chans=(32, 64, 96, 128, 160)):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(1, chans[0], 129, stride=4, padding=64, bias=False),
                                   nn.BatchNorm1d(chans[0]), nn.ReLU(), nn.MaxPool1d(3))
        self.blocks = nn.Sequential(*[StudentBlock(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.fc = nn.Sequential(nn.Linear(2 * chans[-1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):  # (B, N)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        h = self.blocks(self.front(x.unsqueeze(1)))
        h = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return self.fc(h).squeeze(-1)


def build(name, cfg):
    if name == "cnn":
        return MelCNN(cfg["cnn"]["n_mels"])
    if name == "w2v":
        return W2VHead()
    if name == "student":
        return Student()
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
