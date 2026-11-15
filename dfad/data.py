"""Dataset parsing (ASVspoof 2019 LA, In-the-Wild, Hindi) and audio loading.

Label convention everywhere: 0 = real (bona fide), 1 = fake (spoof).
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass
class Item:
    path: str
    label: int
    speaker: str = ""
    attack: str = ""


# ---------------------------------------------------------------- audio

def load_audio(src, sr=SR):
    """Load a file path or file-like object -> float32 mono at `sr`."""
    x, fs = sf.read(src, dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), fs, sr)


def resample(x, fs, sr=SR):
    if fs == sr:
        return x.astype(np.float32)
    g = math.gcd(int(fs), int(sr))
    return resample_poly(x, sr // g, fs // g).astype(np.float32)


def fix_length(x, n, rng=None):
    """Repeat-pad short clips; crop long ones (random crop if rng given, else from the start)."""
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, math.ceil(n / len(x)))
    start = rng.integers(0, len(x) - n + 1) if rng is not None else 0
    return x[start:start + n]


def window_starts(length, n, hop, max_windows=None):
    """Start samples of the windows `windows()` cuts; long clips get `max_windows` spread evenly."""
    if length <= n:
        return [0]
    starts = list(range(0, length - n + 1, hop))
    if max_windows and len(starts) > max_windows:
        starts = np.linspace(0, length - n, max_windows).astype(int).tolist()
    return starts


def windows(x, n, hop, max_windows=None):
    """Split into fixed windows (n samples, `hop` apart). Short clips give one padded window."""
    if len(x) <= n:
        return fix_length(x, n)[None]
    return np.stack([x[s:s + n] for s in window_starts(len(x), n, hop, max_windows)])


# ---------------------------------------------------------------- datasets

def asvspoof_items(root, split):
    """split: train | dev | eval"""
    root = Path(root)
    proto = {"train": "trn", "dev": "trl", "eval": "trl"}[split]
    proto = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{proto}.txt"
    audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
    items = []
    with open(proto, encoding="utf-8") as f:
        for line in f:
            spk, utt, _, attack, key = line.split()
            items.append(Item(str(audio_dir / f"{utt}.flac"), int(key == "spoof"), spk, attack))
    return items


def itw_items(root):
    """In-the-Wild: meta.csv with columns file, speaker, label (bona-fide | spoof)."""
    root = Path(root)
    with open(root / "meta.csv", encoding="utf-8") as f:
        return [Item(str(root / r["file"]), int(r["label"].strip() == "spoof"), r["speaker"])
                for r in csv.DictReader(f)]


def hindi_items(root):
    """hindi/real/** and hindi/fake/** audio files; speaker = first sub-folder name."""
    root = Path(root)
    items = []
    for label, sub in ((0, "real"), (1, "fake")):
        for p in sorted((root / sub).rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                rel = p.relative_to(root / sub).parts
                items.append(Item(str(p), label, rel[0] if len(rel) > 1 else ""))
    return items


DEEPVOICE_TEST_PEOPLE = {"musk", "trump"}
DEEPVOICE_DEV_PEOPLE = {"taylor"}


def deepvoice_items(root, part=None):
    """DEEP-VOICE (Bird & Lotfi 2023): real speeches vs. RVC voice conversions ("Obama-to-Biden").
    part=train/dev/test holds out whole people: a test clip involves a test person (as source or
    target voice), a dev clip involves a dev person but no test person, a train clip involves neither."""
    root = Path(root)
    items = []
    with open(root / "meta.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = Path(r["file"]).stem.split("_frag")[0].lower()
            people = set(stem.replace("-original", "").split("-to-"))
            test, dev = bool(people & DEEPVOICE_TEST_PEOPLE), bool(people & DEEPVOICE_DEV_PEOPLE)
            if (part == "test" and not test) or (part == "dev" and (test or not dev)) \
                    or (part == "train" and (test or dev)):
                continue
            items.append(Item(str(root / "audio" / r["file"]), int(r["label"]), "|".join(sorted(people))))
    return items


ITW_SPLIT = {"train": (0.0, 0.7), "dev": (0.7, 0.8), "test": (0.8, 1.0)}


def speaker_split(items, part, seed=0):
    """Speaker-disjoint split (train 70% / dev 10% / test 20% of speakers): no test speaker is
    ever heard in training, so test EER measures generalisation to new people."""
    speakers = sorted({it.speaker for it in items})
    np.random.default_rng(seed).shuffle(speakers)
    lo, hi = ITW_SPLIT[part]
    keep = set(speakers[int(lo * len(speakers)):int(hi * len(speakers))])
    return [it for it in items if it.speaker in keep]


def get_items(cfg, name):
    """name: asvspoof_train | asvspoof_dev | asvspoof_eval | itw | itw_train | itw_dev | itw_test | hindi"""
    paths = cfg["paths"]
    if name.startswith("asvspoof_"):
        return asvspoof_items(paths["asvspoof_la"], name.split("_", 1)[1])
    if name == "itw":
        return itw_items(paths["in_the_wild"])
    if name.startswith("itw_"):
        return speaker_split(itw_items(paths["in_the_wild"]), name.split("_", 1)[1])
    if name == "hindi":
        return hindi_items(paths["hindi"])
    if name.startswith("deepvoice"):
        return deepvoice_items(paths["deepvoice"], name.split("_", 1)[1] if "_" in name else None)
    raise ValueError(name)


def get_many(cfg, names, limit=0):
    """Concatenate several sets (each optionally subset to `limit`). "name*k" repeats a set k times
    (oversampling: a small but important set is seen k times as often per epoch)."""
    out = []
    for n in names:
        n, _, k = n.partition("*")
        out += subset(get_items(cfg, n), limit) * int(k or 1)
    return out


def subset(items, limit, seed=0):
    """Class-balanced random subset (for smoke runs)."""
    if not limit or limit >= len(items):
        return items
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        pool = [it for it in items if it.label == label]
        k = min(len(pool), limit // 2)
        out += [pool[i] for i in rng.choice(len(pool), k, replace=False)]
    return out


def thread_batches(ds, batch_size, workers=4, prefetch=4):
    """Yield (x, y, idx) batches loaded by a thread pool, in order. Unlike DataLoader worker
    processes this needs no Windows shared-memory handles, which fail outside an interactive
    session (e.g. Task Scheduler). Audio decoding releases the GIL, so threads keep up."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def load(i0):
        rows = [ds[i] for i in range(i0, min(i0 + batch_size, len(ds)))]
        return (torch.stack([r[0] for r in rows]), torch.tensor([r[1] for r in rows]),
                torch.tensor([r[2] for r in rows]))

    starts = iter(range(0, len(ds), batch_size))
    with ThreadPoolExecutor(workers) as ex:
        q = deque(ex.submit(load, s) for s, _ in zip(starts, range(prefetch * workers)))
        while q:
            batch = q.popleft().result()
            nxt = next(starts, None)
            if nxt is not None:
                q.append(ex.submit(load, nxt))
            yield batch


class ClipDataset(Dataset):
    """Fixed-length raw waveforms. Train mode: random crop + augmentation."""

    def __init__(self, items, seconds=4.0, augmenter=None, train=False, seed=0):
        self.items, self.n = items, int(seconds * SR)
        self.augmenter, self.train, self.seed = augmenter, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = load_audio(it.path)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            x = fix_length(x, self.n, rng)
            if self.augmenter is not None:
                x = self.augmenter(x, rng)
        else:
            x = fix_length(x, self.n)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), it.label, i
