"""App-realistic accuracy simulation of the phone model, run exactly the way the apps run it.

  python scripts/simulate_app.py                                # shipped web/model/student.onnx
  python scripts/simulate_app.py --ckpt checkpoints/student.pt  # a checkpoint (before exporting it)

1. "Check a clip" (Android Share / file pick, web Check a clip): each test clip is degraded the
   way it reaches the phone (WhatsApp voice note, phone call, call heard on speakerphone, noisy
   room), cut into 4 s windows every 2 s (max 30), window P(fake) averaged, shipped threshold.
2. "Live call" (Android bubble, web Call Guard live): clips of one speaker are joined into ~45 s
   calls heard over speakerphone; every 2 s the last 4 s are scored (silence skipped by the same
   energy VAD) and the alert fires after `consecutive` red windows in a row.

Test sets are held out from training (unseen speakers / people / attacks). Note that training
used similar degradations, so the conditions here are realistic but not unseen.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dfad import ROOT, load_config
from dfad.augment import add_noise, opus_roundtrip, phone_channel, room_mic
from dfad.data import SR, get_items, load_audio, subset, windows
from dfad.metrics import compute_eer

CONDITIONS = {
    "clean": lambda x, rng: x,
    "voice note (Opus)": opus_roundtrip,
    "phone call (8 kHz)": phone_channel,
    "speakerphone": lambda x, rng: add_noise(room_mic(phone_channel(x, rng), rng), rng, (15, 25)),
    "noisy room (5-10 dB)": lambda x, rng: add_noise(x, rng, (5, 10)),
}
SET_NAMES = {"itw_test": "In-the-Wild (new speakers)", "deepvoice_test": "DEEP-VOICE RVC (new people)",
             "asvspoof_eval": "ASVspoof eval (new TTS attacks)"}


class Scorer:
    """Batch of 4 s windows -> P(fake), from the shipped ONNX file or a PyTorch checkpoint."""

    def __init__(self, onnx=None, ckpt=None, threads=2, cfg=None):
        if ckpt:
            import torch
            from dfad.calibrate import live_threshold
            from dfad.infer import load_checkpoint
            torch.set_num_threads(threads)
            self.net, ck = load_checkpoint(ckpt, "cpu")
            self.threshold, self.name = float(ck["threshold"]), Path(ckpt).name
            self._run = lambda w: torch.sigmoid(self.net(torch.from_numpy(w))).detach().numpy()
            with torch.no_grad():  # what export_onnx.py would ship as live_threshold
                self.live_threshold = live_threshold(self.probs, cfg, self.threshold)[0]
        else:
            import onnxruntime as ort
            onnx = Path(onnx)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
            conf = json.loads((onnx.parent / "config.json").read_text())
            self.threshold = conf["threshold"]
            self.live_threshold = conf.get("live_threshold", self.threshold)
            self.name = onnx.name
            self._run = lambda w: sess.run(None, {"audio": w})[0]

    def probs(self, wins, batch=32):
        wins = np.ascontiguousarray(wins, dtype=np.float32)
        return np.concatenate([self._run(wins[i:i + batch]) for i in range(0, len(wins), batch)])


def has_speech(x, f=320):
    """Same energy VAD as web/app.js and Detector.kt: >25% of 20 ms frames above -45 dBFS."""
    e = (x[:len(x) // f * f].reshape(-1, f).astype(np.float64) ** 2).mean(1)
    return len(e) > 0 and (10 * np.log10(e + 1e-12) > -45).mean() > 0.25


def degrade(cond, x, seed):
    return np.clip(CONDITIONS[cond](x, np.random.default_rng(seed)), -1, 1).astype(np.float32)


def clip_check(scorer, items, audio, n, hop):
    rows = []
    for cond in CONDITIONS:
        p = np.array([scorer.probs(windows(degrade(cond, audio[i], i), n, hop, 30)).mean()
                      for i in range(len(items))])
        y = np.array([it.label for it in items])
        fake = p >= scorer.threshold
        rows.append({"condition": cond, "eer": compute_eer(y, p)[0],
                     "accuracy": ((fake == y.astype(bool)).astype(float)[y == 0].mean()
                                  + (fake == y.astype(bool)).astype(float)[y == 1].mean()) / 2,
                     "false_alarm": fake[y == 0].mean(), "caught": fake[y == 1].mean()})
        print(f"  {cond:22s} acc {rows[-1]['accuracy']:.1%}  real flagged {rows[-1]['false_alarm']:.1%}  "
              f"fakes caught {rows[-1]['caught']:.1%}  EER {rows[-1]['eer']:.1%}", flush=True)
    return rows


def make_calls(items, per_label, seconds, rng):
    """Join clips of the same speaker (and label) into ~`seconds` calls, with short pauses."""
    groups = defaultdict(list)
    for it in items:
        groups[(it.speaker, it.label)].append(it)
    calls = []
    for label in (0, 1):
        keys = [k for k in groups if k[1] == label]
        for c in range(per_label):
            if not keys:
                break
            pool = groups[keys[c % len(keys)]]
            parts, total = [], 0
            for j in rng.permutation(len(pool)):
                x = load_audio(pool[j].path)
                gap = (rng.standard_normal(int(rng.uniform(0.3, 1.0) * SR)) * 1e-4).astype(np.float32)
                parts += [x, gap]
                total += len(x) + len(gap)
                if total >= seconds * SR:
                    break
            if total >= 12 * SR:  # at least a few windows
                calls.append((label, np.concatenate(parts)))
    return calls


def live_calls(scorer, calls, n, hop, consecutive):
    res = {0: [], 1: []}
    for k, (label, x) in enumerate(calls):
        x = degrade("speakerphone", x, 10_000 + k)
        ends = list(range(n, len(x) + 1, hop))
        rings = [x[e - n:e] for e in ends]
        scored = [i for i, r in enumerate(rings) if np.abs(r).max() > 0 and has_speech(r)]
        p = dict(zip(scored, scorer.probs(np.stack([rings[i] for i in scored])))) if scored else {}
        streak, alert_at, red = 0, None, 0
        for i in scored:  # windows without speech leave the streak unchanged, as in the apps
            is_red = p[i] >= scorer.live_threshold
            red += is_red
            streak = streak + 1 if is_red else 0
            if streak == consecutive and alert_at is None:
                alert_at = ends[i] / SR
        res[label].append({"alert_at": alert_at, "red_frac": red / max(1, len(scored)), "sec": len(x) / SR})
    out = {}
    for label, r in res.items():
        if not r:
            continue
        t = [c["alert_at"] for c in r if c["alert_at"] is not None]
        out[label] = {"calls": len(r), "alerted": len(t) / len(r), "median_alert_s": float(np.median(t)) if t else None,
                      "red_windows": float(np.mean([c["red_frac"] for c in r])),
                      "avg_len_s": float(np.mean([c["sec"] for c in r]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "web" / "model" / "student.onnx"))
    ap.add_argument("--ckpt", default=None, help="score a PyTorch checkpoint instead of the ONNX file")
    ap.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    ap.add_argument("--per-set", type=int, default=400, help="clips per set (half real, half fake)")
    ap.add_argument("--calls", type=int, default=30, help="simulated calls per label per set")
    ap.add_argument("--call-seconds", type=int, default=45)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n, hop = int(cfg["audio"]["seconds"] * SR), int(cfg["audio"]["hop_seconds"] * SR)
    scorer = Scorer(args.onnx, args.ckpt, args.threads, cfg)
    print(f"model {scorer.name}  threshold {scorer.threshold:.4f}  live {scorer.live_threshold:.4f}", flush=True)

    md = [f"# App simulation: `{scorer.name}` (threshold {scorer.threshold:.4f}, live-call threshold "
          f"{scorer.live_threshold:.4f})", "",
          "Accuracy = balanced (mean of real-correct and fake-correct). Real flagged = false alarms on "
          "genuine voices. EER is threshold-free (how separable the scores are).", ""]
    t0 = time.time()
    for s in args.sets:
        try:
            items = subset(get_items(cfg, s), args.per_set)
        except (FileNotFoundError, OSError) as e:
            print(f"skip {s}: {e}")
            continue
        print(f"\n== {SET_NAMES.get(s, s)}: {len(items)} clips", flush=True)
        audio = [load_audio(it.path) for it in items]
        rows = clip_check(scorer, items, audio, n, hop)
        md += [f"## {SET_NAMES.get(s, s)}", "", "**Check a clip** "
               f"({sum(it.label == 0 for it in items)} real / {sum(it.label == 1 for it in items)} fake clips)", "",
               "| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |",
               "|---|---|---|---|---|"]
        md += [f"| {r['condition']} | {r['accuracy']:.1%} | {r['false_alarm']:.1%} | {r['caught']:.1%} | {r['eer']:.1%} |"
               for r in rows]

        calls = make_calls(get_items(cfg, s), args.calls, args.call_seconds, np.random.default_rng(0))
        live = live_calls(scorer, calls, n, hop, 3)
        md += ["", f"**Live call on speakerphone** (~{args.call_seconds} s calls, alert after 3 red windows in a row)", "",
               "| caller | calls | alert raised | median time to alert | red windows |", "|---|---|---|---|---|"]
        for label, name in ((0, "real person"), (1, "AI voice")):
            if label in live:
                L = live[label]
                when = f"{L['median_alert_s']:.0f} s" if L["median_alert_s"] else "-"
                md.append(f"| {name} | {L['calls']} | {L['alerted']:.0%} | {when} | {L['red_windows']:.0%} |")
                print(f"  live {name:12s} calls {L['calls']}  alerted {L['alerted']:.0%}  "
                      f"median at {when}  red windows {L['red_windows']:.0%}", flush=True)
        md.append("")

    out = Path(cfg["paths"]["results"]) / f"app_simulation_{Path(scorer.name).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
