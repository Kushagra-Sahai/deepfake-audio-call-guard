"""Live-call threshold. The checkpoint threshold is set on clean dev clips, but during a live call
the mic hears the caller through the phone's loudspeaker, which pushes every score up, genuine
voices included. So live checking gets its own threshold: calibrated on genuine dev clips run
through the speakerphone chain, so that only `target_far` of their windows turn red.

15% was chosen with a sweep on simulated calls: it alerts on 7-10% of genuine callers while
catching 73-80% of AI callers; stricter targets missed many voice-changer calls, looser ones
roughly doubled the false alarms."""
import numpy as np

from .augment import speakerphone
from .data import SR, fix_length, get_many, load_audio


def live_threshold(probs, cfg, clean_threshold, per_set=400, target_far=0.15):
    """probs: (W, N) float32 windows -> (W,) P(fake). Returns (threshold, n_windows); the
    threshold is never below the clean one."""
    n = int(cfg["audio"]["seconds"] * SR)
    sets = [s.partition("*")[0] for s in cfg["student"]["dev_sets"]]
    real = [it for it in get_many(cfg, sets, per_set) if it.label == 0]
    wins = np.stack([speakerphone(fix_length(load_audio(it.path), n), np.random.default_rng(i))
                     for i, it in enumerate(real)]).astype(np.float32)
    p = np.concatenate([probs(wins[i:i + 32]) for i in range(0, len(wins), 32)])
    return max(float(np.quantile(p, 1 - target_far)), float(clean_threshold)), len(p)
