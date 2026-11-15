"""Export the Student to ONNX for the phone (web + Android), optionally int8, with a parity check.

  python scripts/export_onnx.py            # fp32 (~0.8 MB)
  python scripts/export_onnx.py --int8     # static int8 QDQ, calibrated on ASVspoof dev clips

Writes web/model/{student.onnx, config.json} and copies both into the Android app assets.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from dfad import ROOT, load_config
from dfad.calibrate import live_threshold
from dfad.data import fix_length, get_items, load_audio, subset
from dfad.infer import load_checkpoint

WEB = ROOT / "web" / "model"
ANDROID = ROOT / "android" / "app" / "src" / "main" / "assets"


class ProbWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, audio):
        return torch.sigmoid(self.net(audio))


def sample_windows(cfg, n, k):
    try:
        items = subset(get_items(cfg, "asvspoof_dev"), k)
        return np.stack([fix_length(load_audio(it.path), n) for it in items]).astype(np.float32)
    except (FileNotFoundError, OSError):
        print("dataset not found -> using noise for calibration/parity (less representative)")
        return (np.random.default_rng(0).standard_normal((k, n)) * 0.1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    net, ck = load_checkpoint(args.ckpt or Path(cfg["paths"]["checkpoints"]) / "student.pt")
    assert ck["model"] == "student", "only the Student is small enough for phones"
    n = int(cfg["audio"]["seconds"] * 16000)
    model = ProbWrapper(net).eval()

    WEB.mkdir(parents=True, exist_ok=True)
    fp32 = WEB / "student_fp32.onnx"
    kw = dict(input_names=["audio"], output_names=["prob"], opset_version=17,
              dynamic_axes={"audio": {0: "batch"}, "prob": {0: "batch"}})
    try:  # newer torch defaults to the dynamo exporter; the classic one is simpler here
        torch.onnx.export(model, torch.zeros(1, n), fp32, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, torch.zeros(1, n), fp32, **kw)

    x = sample_windows(cfg, n, 64)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = WEB / "student.onnx"
    chosen = fp32
    if args.int8:
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                              quantize_static)
        from onnxruntime.quantization.shape_inference import quant_pre_process

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"audio": w[None]} for w in x])

            def get_next(self):
                return next(self.it, None)

        pre = WEB / "student_pre.onnx"
        quant_pre_process(str(fp32), str(pre))
        q = WEB / "student_int8.onnx"
        quantize_static(str(pre), str(q), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
        pre.unlink()
        diff = np.abs(ort.InferenceSession(str(q)).run(None, {"audio": x})[0] - ref).max()
        print(f"int8 max |prob diff| vs PyTorch = {diff:.4f}")
        if diff < 0.05:
            chosen = q
        else:
            print("int8 parity too poor -> shipping fp32")

    shutil.copy(chosen, out)
    diff = np.abs(ort.InferenceSession(str(out)).run(None, {"audio": x})[0] - ref).max()
    print(f"shipped {chosen.name}: {out.stat().st_size / 1e6:.2f} MB, max |prob diff| {diff:.4f}")

    thr = float(ck["threshold"])
    try:
        with torch.no_grad():
            live, k = live_threshold(lambda w: model(torch.from_numpy(w)).numpy(), cfg, thr)
        print(f"live-call threshold {live:.4f} (speakerphone-degraded genuine dev windows: {k})")
    except (FileNotFoundError, OSError) as e:
        live = thr
        print(f"dev sets not found ({e}) -> live threshold = clean threshold")
    conf = {"sr": 16000, "window": n, "hop": int(cfg["audio"]["hop_seconds"] * 16000),
            "threshold": round(thr, 4), "live_threshold": round(live, 4),
            "consecutive": 3, "dev_eer": round(float(ck["dev_eer"]), 4)}
    (WEB / "config.json").write_text(json.dumps(conf, indent=2))
    ANDROID.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, ANDROID / "student.onnx")
    shutil.copy(WEB / "config.json", ANDROID / "config.json")
    print(json.dumps(conf))


if __name__ == "__main__":
    main()
