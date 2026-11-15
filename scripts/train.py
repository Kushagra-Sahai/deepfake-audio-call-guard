"""Train one model on ASVspoof 2019 LA train, select on dev EER.

  python scripts/train.py --model gmm
  python scripts/train.py --model cnn
  python scripts/train.py --model w2v        # needs: cache_xlsr.py asvspoof_train asvspoof_dev
  add --limit 500 for a quick smoke run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dfad import load_config
from dfad.augment import Augmenter
from dfad.cache import CachedFeats
from dfad.data import ClipDataset, fix_length, get_items, load_audio, subset
from dfad.metrics import compute_eer
from dfad.models import GMMClassifier, W2VHead, build, mfcc_frames, n_params
from dfad.trainer import bce, fit, pos_weight, seed_all


def train_gmm(cfg, args, ckpt_dir):
    g = cfg["gmm"]
    n = int(cfg["audio"]["seconds"] * 16000)
    train = subset(get_items(cfg, "asvspoof_train"), args.limit or 6000)
    frames = {0: [], 1: []}
    for it in tqdm(train, desc="MFCC"):
        frames[it.label].append(mfcc_frames(load_audio(it.path), g["n_mfcc"]))
    rng = np.random.default_rng(0)
    for k in frames:
        f = np.concatenate(frames[k])
        if len(f) > g["max_frames_per_class"]:
            f = f[rng.choice(len(f), g["max_frames_per_class"], replace=False)]
        frames[k] = f
    n_comp = min(g["n_components"], len(frames[0]) // 50)
    print(f"fitting 2 GMMs x {n_comp} components on {len(frames[0])}/{len(frames[1])} frames")
    gmm = GMMClassifier(n_comp, g["n_mfcc"]).fit(frames[0], frames[1])
    dev = subset(get_items(cfg, "asvspoof_dev"), args.limit)
    scores = [gmm.score(fix_length(load_audio(it.path), n)) for it in tqdm(dev, desc="dev")]
    eer, gmm.threshold = compute_eer([it.label for it in dev], scores)
    gmm.save(ckpt_dir / "gmm.joblib")
    print(f"GMM dev EER {eer * 100:.2f}%")


def cached_sets(cfg, names, copies=None):
    """Concatenate cached XLS-R feature sets; "name*k" repeats a set k times (oversampling)."""
    parts = []
    for n in names:
        n, _, k = n.partition("*")
        parts += [CachedFeats(cfg, n, copies=copies)] * int(k or 1)
    labels = np.concatenate([np.tile(p.y[p.idx], p.copies) for p in parts])
    return ConcatDataset(parts), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gmm", "cnn", "w2v"])
    ap.add_argument("--limit", type=int, default=0, help="class-balanced subset size (smoke runs)")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["train"]["seed"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = cfg["train"]["num_workers"]

    if args.model == "gmm":
        return train_gmm(cfg, args, ckpt_dir)

    hp = cfg[args.model]
    if args.model == "cnn":
        tr_items = subset(get_items(cfg, "asvspoof_train"), args.limit)
        dv_items = subset(get_items(cfg, "asvspoof_dev"), args.limit)
        aug = Augmenter(cfg["augment"], cfg["paths"]["noise_dir"])
        tr = ClipDataset(tr_items, cfg["audio"]["seconds"], aug, train=True, seed=cfg["train"]["seed"])
        dv = ClipDataset(dv_items, cfg["audio"]["seconds"])
        labels = [it.label for it in tr_items]
        net = build("cnn", cfg)
    else:  # subset sizes for w2v are chosen when caching (cache_xlsr.py --limit)
        tr, labels = cached_sets(cfg, hp.get("train_sets", ["asvspoof_train"]))
        dv, _ = cached_sets(cfg, hp.get("dev_sets", ["asvspoof_dev"]), copies=1)
        f0 = tr[0][0]
        net = W2VHead(n_layers=f0.shape[0], dim=f0.shape[1])
        workers = 0  # memory-mapped features: worker processes would copy the whole array

    print(f"{args.model}: {n_params(net) / 1e6:.2f}M trainable params, {len(tr)} train / {len(dv)} dev")
    tl = DataLoader(tr, hp["batch_size"], shuffle=True, num_workers=workers, drop_last=True,
                    persistent_workers=False)  # fit() changes dataset.epoch; persistent workers would never see it
    dl = DataLoader(dv, hp["batch_size"] * 2, num_workers=workers)
    best = fit(net, tl, dl, epochs=args.epochs or hp["epochs"], lr=hp["lr"], device=device,
               ckpt_path=ckpt_dir / f"{args.model}.pt", meta={"model": args.model, "cfg": cfg},
               loss_fn=bce(pos_weight(labels, device)))
    print(f"best dev EER {best * 100:.2f}% -> {ckpt_dir / (args.model + '.pt')}")


if __name__ == "__main__":
    main()
