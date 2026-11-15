"""Equal Error Rate. Positive class = fake (label 1); higher score = more likely fake."""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    """Return (eer, threshold). The threshold is on the same scale as `scores`."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise ValueError("EER needs both real and fake samples")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    # interpolate between the two points around the crossing
    j = i + 1 if i + 1 < len(fpr) and (fnr[i] - fpr[i]) > 0 else i - 1
    j = min(max(j, 0), len(fpr) - 1)
    d_i, d_j = fnr[i] - fpr[i], fnr[j] - fpr[j]
    if j != i and d_i != d_j:
        w = d_i / (d_i - d_j)
        eer = fpr[i] + w * (fpr[j] - fpr[i])
    else:
        eer = (fpr[i] + fnr[i]) / 2
    t = thr[i] if np.isfinite(thr[i]) else scores.max()
    return float(eer), float(t)
