"""dfad: deepfake audio detection."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("hindi", "cache", "checkpoints", "results", "noise_dir"):
        p = cfg["paths"].get(key)
        if p and not Path(p).is_absolute():
            cfg["paths"][key] = str(ROOT / p)
    return cfg
