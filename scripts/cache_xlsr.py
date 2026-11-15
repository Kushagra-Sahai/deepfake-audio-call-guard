"""Run frozen XLS-R once over a dataset and cache pooled layer features (fp16) to disk.

  python scripts/cache_xlsr.py asvspoof_train asvspoof_dev asvspoof_eval itw [hindi]

Train sets also get `w2v.train_aug_copies` augmented copies (the backbone is frozen, so
augmentation happens here instead of on the fly). Resumable: finished chunks are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from dfad import load_config
from dfad.augment import Augmenter
from dfad.cache import feat_path, files_path, labels_path
from dfad.data import ClipDataset, get_items, subset, thread_batches
from dfad.models import XLSRFeatures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="class-balanced subset size (smoke runs)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = XLSRFeatures(cfg["w2v"]["model_name"]).to(device)
    if device == "cuda":
        net.half()
    aug = Augmenter(cfg["augment"], cfg["paths"]["noise_dir"])

    for name in args.datasets:
        items = subset(get_items(cfg, name), args.limit)
        copies = 1 + (cfg["w2v"]["train_aug_copies"] if name.endswith("train") else 0)
        out = feat_path(cfg, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path(cfg, name), np.array([it.label for it in items], np.int8))
        files_path(cfg, name).write_text("\n".join(it.path for it in items), encoding="utf-8")
        done_file = out.with_suffix(".done")
        done = set(done_file.read_text().split()) if done_file.exists() else set()
        with torch.no_grad():
            probe = net(torch.zeros(1, 16000, device=device, dtype=next(net.parameters()).dtype))
        shape = (copies, len(items), *probe.shape[1:])
        mode = "r+" if out.exists() and done else "w+"
        feats = np.lib.format.open_memmap(out, mode=mode, dtype=np.float16, shape=shape)
        for c in range(copies):
            if str(c) in done:
                continue
            ds = ClipDataset(items, cfg["audio"]["seconds"], aug if c > 0 else None, train=c > 0, seed=c)
            bs = cfg["w2v"]["cache_batch_size"]
            dl = thread_batches(ds, bs, workers=cfg["train"]["num_workers"])
            for x, _, idx in tqdm(dl, desc=f"{name} copy {c}", total=-(-len(ds) // bs), mininterval=30):
                x = x.to(device)
                with torch.no_grad():
                    f = net(x.half() if device == "cuda" else x)
                feats[c, idx.numpy()] = f.float().cpu().numpy().astype(np.float16)
            feats.flush()
            done.add(str(c))
            done_file.write_text(" ".join(sorted(done)))
        print(f"{name}: {shape} -> {out}")


if __name__ == "__main__":
    main()
