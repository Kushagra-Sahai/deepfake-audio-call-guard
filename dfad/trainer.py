"""Shared training loop: AdamW + cosine LR, AMP on GPU, keep the checkpoint with the best dev EER."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_eer


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pos_weight(labels, device):
    """ASVspoof is ~1 real : 9 fake, so down-weight the fake (positive) class."""
    labels = np.asarray(labels)
    return torch.tensor((labels == 0).sum() / max((labels == 1).sum(), 1), device=device)


@torch.no_grad()
def predict(net, loader, device, to_input=lambda b: b[0]):
    net.eval()
    labels, logits = [], []
    for b in loader:
        with torch.autocast(device, enabled=device == "cuda"):
            logits.append(net(to_input(b).to(device)).float().cpu())
        labels.append(b[1])
    return torch.cat(labels).numpy(), torch.cat(logits).numpy()


def fit(net, train_loader, dev_loader, *, epochs, lr, device, ckpt_path, meta, loss_fn,
        to_input=lambda b: b[0], on_best=None, best=float("inf")):
    """loss_fn(logits, batch) -> scalar. Saves {**meta, state, threshold, dev_eer} at best dev EER,
    then calls on_best() (e.g. to export the improved model straight away)."""
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    for ep in range(epochs):
        if hasattr(train_loader.dataset, "epoch"):
            train_loader.dataset.epoch = ep
        net.train()
        t0, run = time.time(), 0.0
        for b in train_loader:
            with torch.autocast(device, enabled=device == "cuda"):
                logits = net(to_input(b).to(device))
            loss = loss_fn(logits.float(), b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        labels, logits = predict(net, dev_loader, device, to_input)
        eer, thr = compute_eer(labels, logits)
        flag = ""
        if eer < best:
            best, flag = eer, "  *saved*"
            torch.save({**meta, "state": net.state_dict(), "dev_eer": eer,
                        "threshold": float(1 / (1 + math.exp(-thr))), "epoch": ep}, ckpt_path)
        print(f"epoch {ep + 1}/{epochs}  loss {run / len(train_loader):.4f}  "
              f"dev EER {eer * 100:.2f}%  ({time.time() - t0:.0f}s){flag}", flush=True)
        if flag and on_best:
            on_best()
    return best


def bce(pw):
    def loss(logits, b):
        return F.binary_cross_entropy_with_logits(logits, b[1].float().to(logits.device), pos_weight=pw)
    return loss
