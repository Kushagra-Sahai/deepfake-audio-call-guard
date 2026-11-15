"""Build the fake half of the Hindi test set by cloning volunteer voices with Coqui XTTS-v2.

Input  (collect with written consent):
  data/hindi/real/<speaker>/*.wav      volunteer recordings (16 kHz+, 3-10 s each)
  data/hindi/sentences.txt             one Hindi sentence per line (Devanagari)
Output:
  data/hindi/fake/<speaker>/xtts_<k>.wav   + data/hindi/meta.csv

XTTS-v2 is heavy (~2 GB, GPU recommended); run in a separate env / Colab:
  pip install coqui-tts
  python scripts/make_hindi_fakes.py --per-speaker 20
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from dfad import load_config
from dfad.data import SR, hindi_items, resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["hindi"])
    sentences = [s.strip() for s in (root / "sentences.txt").read_text(encoding="utf-8").splitlines() if s.strip()]

    import torch
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")

    real = [it for it in hindi_items(root) if it.label == 0]
    speakers = sorted({it.speaker for it in real})
    rng = np.random.default_rng(0)
    rows = [{"file": str(Path(it.path).relative_to(root)), "speaker": it.speaker, "label": "real", "tool": ""}
            for it in real]
    for spk in speakers:
        refs = [it.path for it in real if it.speaker == spk][:3]  # XTTS accepts several reference clips
        out_dir = root / "fake" / spk
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(rng.choice(len(sentences), min(args.per_speaker, len(sentences)), replace=False)):
            wav = np.asarray(tts.tts(text=sentences[i], speaker_wav=refs, language="hi"), np.float32)
            out = out_dir / f"xtts_{k:03d}.wav"
            sf.write(out, resample(wav, tts.synthesizer.output_sample_rate, SR), SR)
            rows.append({"file": str(out.relative_to(root)), "speaker": spk, "label": "fake", "tool": "xtts_v2"})
        print(f"{spk}: {k + 1} clips")
    with open(root / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["file", "speaker", "label", "tool"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
