"""Distil the XLS-R teacher (w2v.pt) into the tiny raw-audio Student for phones.

  python scripts/distill.py              # needs checkpoints/w2v.pt + cached student.train_sets
  python scripts/distill.py --no-teacher # plain supervised training of the Student

Loss = (1 - a) * BCE(label) + a * BCE(teacher soft label at temperature T).
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dfad import load_config
from dfad.augment import Augmenter
from dfad.cache import CachedFeats, cached_items, is_cached
from dfad.data import ClipDataset, get_many
from dfad.infer import load_checkpoint
from dfad.models import Student, n_params
from dfad.trainer import fit, pos_weight, predict, seed_all


def teacher_targets(cfg, names, device):
    """For every training set (in cache order, "name*k" repeated k times): the clip list and the
    XLS-R teacher's logit on its clean first 4 s (the student sees a random crop + augmentation
    of the same clip, so the soft label is a per-clip target, not a per-crop one)."""
    head, _ = load_checkpoint(Path(cfg["paths"]["checkpoints"]) / "w2v.pt", device)
    items, logits = [], []
    for n in names:
        n, _, k = n.partition("*")
        if not is_cached(cfg, n):
            sys.exit(f"no cached XLS-R features for {n}; run cache_xlsr.py {n} or pass --no-teacher")
        _, lg = predict(head, DataLoader(CachedFeats(cfg, n, copies=1), 512), device)
        items += cached_items(cfg, n) * int(k or 1)
        logits += [lg] * int(k or 1)
    return items, torch.tensor(np.concatenate(logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from this student checkpoint (fine-tune)")
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--keep-best", action="store_true",
                    help="with --init: never overwrite the model with a round worse than it")
    ap.add_argument("--export-each", action="store_true",
                    help="export to web/ + android/ every time the dev EER improves")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp, sec, workers = cfg["student"], cfg["audio"]["seconds"], cfg["train"]["num_workers"]

    use_teacher = not args.no_teacher
    if use_teacher:
        tr_items, t_logits = teacher_targets(cfg, hp.get("train_sets", ["asvspoof_train"]), device)
        print(f"teacher logits for {len(t_logits)} clips")
    else:
        tr_items = get_many(cfg, hp.get("train_sets", ["asvspoof_train"]), args.limit)
    dv_items = get_many(cfg, hp.get("dev_sets", ["asvspoof_dev"]), args.limit or hp.get("dev_limit", 0))

    aug = Augmenter(cfg["augment"], cfg["paths"]["noise_dir"])
    tr = ClipDataset(tr_items, sec, aug, train=True, seed=cfg["train"]["seed"])
    dv = ClipDataset(dv_items, sec)
    net = Student()
    best0 = float("inf")
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        net.load_state_dict(init["state"])
        if args.keep_best:  # only save rounds that beat the starting checkpoint
            best0 = init["dev_eer"]
        print(f"fine-tuning from {args.init} (its dev EER {init['dev_eer'] * 100:.2f}%)")
    print(f"student: {n_params(net) / 1e6:.3f}M params, {len(tr)} train / {len(dv)} dev")

    pw = pos_weight([it.label for it in tr_items], device)
    a, T = hp["kd_alpha"], hp["kd_temperature"]

    def loss_fn(logits, b):
        y = b[1].float().to(device)
        hard = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        if not use_teacher:
            return hard
        soft_t = torch.sigmoid(t_logits[b[2]].to(device) / T)
        soft = F.binary_cross_entropy_with_logits(logits / T, soft_t) * T * T
        return (1 - a) * hard + a * soft

    ckpt = Path(cfg["paths"]["checkpoints"]) / "student.pt"
    tl = DataLoader(tr, hp["batch_size"], shuffle=True, num_workers=workers, drop_last=True,
                    persistent_workers=False)  # fit() changes dataset.epoch; persistent workers would never see it
    dl = DataLoader(dv, hp["batch_size"] * 2, num_workers=workers)
    def export():
        subprocess.run([sys.executable, str(Path(__file__).parent / "export_onnx.py")]
                       + (["--config", args.config] if args.config else []), check=False)

    best = fit(net, tl, dl, epochs=args.epochs or hp["epochs"], lr=args.lr or hp["lr"], device=device,
               ckpt_path=ckpt, meta={"model": "student", "cfg": cfg}, loss_fn=loss_fn,
               on_best=export if args.export_each else None, best=best0)
    print(f"best dev EER {best * 100:.2f}% -> {ckpt}")


if __name__ == "__main__":
    main()
