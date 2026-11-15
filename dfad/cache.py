"""On-disk cache of frozen XLS-R features: cache/xlsr/<dataset>.npy, shape (copies, N, layers, 2*dim), fp16."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def feat_path(cfg, name):
    return Path(cfg["paths"]["cache"]) / "xlsr" / f"{name}.npy"


def labels_path(cfg, name):
    return Path(cfg["paths"]["cache"]) / "xlsr" / f"{name}_labels.npy"


def files_path(cfg, name):
    return Path(cfg["paths"]["cache"]) / "xlsr" / f"{name}_files.txt"


def is_cached(cfg, name):
    return all(p.exists() for p in (feat_path(cfg, name), labels_path(cfg, name), files_path(cfg, name)))


def cached_items(cfg, name):
    """The exact clip list (same order) that was cached, so other models can align with it."""
    from .data import Item
    paths = files_path(cfg, name).read_text(encoding="utf-8").splitlines()
    labels = np.load(labels_path(cfg, name))
    return [Item(p, int(l)) for p, l in zip(paths, labels)]


class CachedFeats(Dataset):
    """Yields (features, label, clip_index). With copies > 1 every augmented copy is a sample."""

    def __init__(self, cfg, name, copies=None, limit_idx=None):
        self.x = np.load(feat_path(cfg, name), mmap_mode="r")
        self.y = np.load(labels_path(cfg, name))
        self.copies = self.x.shape[0] if copies is None else min(copies, self.x.shape[0])
        self.idx = np.arange(self.x.shape[1]) if limit_idx is None else np.asarray(limit_idx)

    def __len__(self):
        return self.copies * len(self.idx)

    def __getitem__(self, i):
        c, j = divmod(i, len(self.idx))
        j = self.idx[j]
        return torch.from_numpy(np.array(self.x[c, j], dtype=np.float32)), int(self.y[j]), int(j)
