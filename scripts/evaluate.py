"""EER of every trained model on every available test set -> results/table.csv (+ markdown).

  python scripts/evaluate.py                         # asvspoof_eval, itw_test, deepvoice_test, hindi (whatever exists)
  python scripts/evaluate.py --sets itw --limit 2000

Each clip is scored on its first 4 s (the ASVspoof convention). The XLS-R model uses the
feature cache when present (cache_xlsr.py <set>), otherwise runs the backbone.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dfad import load_config
from dfad.cache import CachedFeats, cached_items, is_cached
from dfad.data import ClipDataset, fix_length, get_items, load_audio, subset
from dfad.infer import load_checkpoint
from dfad.metrics import compute_eer
from dfad.models import GMMClassifier, XLSRFeatures
from dfad.trainer import predict

MODELS = ["gmm", "cnn", "w2v", "student"]


def score(model, cfg, items, set_name, ckpt_dir, device, workers):
    n = int(cfg["audio"]["seconds"] * 16000)
    if model == "gmm":
        gmm = GMMClassifier.load(ckpt_dir / "gmm.joblib")
        return np.array([gmm.score(fix_length(load_audio(it.path), n)) for it in tqdm(items, desc="gmm")])
    net, _ = load_checkpoint(ckpt_dir / f"{model}.pt", device)
    if model == "w2v" and is_cached(cfg, set_name):
        _, s = predict(net, DataLoader(CachedFeats(cfg, set_name, copies=1), 512), device)
        return s
    dl = DataLoader(ClipDataset(items, cfg["audio"]["seconds"]), 32, num_workers=workers)
    if model == "w2v":
        bb = XLSRFeatures(cfg["w2v"]["model_name"]).to(device)
        full = torch.nn.Sequential(bb, net).eval()
        return predict(full, tqdm(dl, desc="w2v"), device)[1]
    return predict(net, tqdm(dl, desc=model), device)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["asvspoof_eval", "itw_test", "deepvoice_test", "hindi"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir, res_dir = Path(cfg["paths"]["checkpoints"]), Path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for set_name in args.sets:
        try:
            if is_cached(cfg, set_name):  # keep the clip order identical to the cache
                items = cached_items(cfg, set_name)
            else:
                items = subset(get_items(cfg, set_name), args.limit)
        except (FileNotFoundError, OSError) as e:
            print(f"skip {set_name}: {e}")
            continue
        if not items:
            print(f"skip {set_name}: empty")
            continue
        labels = np.array([it.label for it in items])
        for model in args.models:
            if not (ckpt_dir / ("gmm.joblib" if model == "gmm" else f"{model}.pt")).exists():
                continue
            s = score(model, cfg, items, set_name, ckpt_dir, device, cfg["train"]["num_workers"])
            eer, _ = compute_eer(labels, s)
            np.save(res_dir / f"scores_{model}_{set_name}.npy", s)
            rows.append({"model": model, "dataset": set_name, "eer_percent": round(eer * 100, 2),
                         "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum())})
            print(rows[-1], flush=True)

    with open(res_dir / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["model", "dataset", "eer_percent", "n_real", "n_fake"])
        w.writeheader()
        w.writerows(rows)
    sets = sorted({r["dataset"] for r in rows}, key=args.sets.index)
    md = ["| model | " + " | ".join(f"{s} EER %" for s in sets) + " |", "|---" * (len(sets) + 1) + "|"]
    for m in [m for m in MODELS if any(r["model"] == m for r in rows)]:
        cells = [next((str(r["eer_percent"]) for r in rows if r["model"] == m and r["dataset"] == s), "-")
                 for s in sets]
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    (res_dir / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
