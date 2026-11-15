import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import soundfile as sf
import torch

from dfad import load_config
from dfad.augment import MP3_OK, Augmenter, mp3_roundtrip, phone_channel
from dfad.data import SR, fix_length, hindi_items, load_audio, window_starts, windows
from dfad.metrics import compute_eer
from dfad.models import GMMClassifier, MelCNN, Student, W2VHead, mfcc_frames, n_params

N = 4 * SR
rng = np.random.default_rng(0)


def tone(sec=1.0, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_load_resamples_to_16k_mono():
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone(1, 44100)] * 2, 1), 44100, format="WAV")
    buf.seek(0)
    x = load_audio(buf)
    assert x.dtype == np.float32 and x.ndim == 1 and abs(len(x) - SR) <= 1


def test_fix_length_and_windows():
    assert fix_length(tone(1), N).shape == (N,)          # repeat-pad
    assert fix_length(tone(6), N, rng).shape == (N,)     # random crop
    w = windows(tone(10), N, 2 * SR)
    assert w.shape == (4, N)
    assert windows(tone(1), N, 2 * SR).shape == (1, N)
    assert windows(tone(60), N, 2 * SR, max_windows=5).shape == (5, N)
    starts = window_starts(60 * SR, N, 2 * SR, max_windows=5)
    assert starts[0] == 0 and starts[-1] == 60 * SR - N  # long clips: spread over the whole clip


def test_risk_maps_threshold_to_half():
    from dfad.infer import risk
    assert risk(0.9386, 0.9386) == pytest.approx(0.5)
    assert risk(0.0, 0.9) == 0 and risk(1.0, 0.9) == pytest.approx(1)
    assert risk(0.9, 0.9386) < 0.5  # below threshold -> "real" side, even though p looks high


def test_augmentations_keep_length_and_change_signal():
    x = fix_length(tone(1), N)
    y = phone_channel(x, rng)
    assert y.shape == x.shape and not np.allclose(x, y)
    if MP3_OK:
        z = mp3_roundtrip(x, rng)
        assert z.shape == x.shape and not np.allclose(x, z)
    aug = Augmenter({**load_config()["augment"], "p": 1.0})
    a = aug(x, rng)
    assert a.shape == x.shape and a.dtype == np.float32 and np.abs(a).max() <= 1


def test_eer():
    labels = np.r_[np.zeros(500), np.ones(500)]
    assert compute_eer(labels, labels + 0.1)[0] == pytest.approx(0, abs=1e-6)
    eer, _ = compute_eer(labels, np.random.default_rng(1).random(1000))
    assert 0.4 < eer < 0.6
    # known overlap: real ~ U(0,1), fake ~ U(0.5,1.5): FPR = 1-t, FNR = t-0.5 -> EER 0.25 at t=0.75
    s = np.r_[np.linspace(0, 1, 3000), np.linspace(0.5, 1.5, 3000)]
    eer, t = compute_eer(np.r_[np.zeros(3000), np.ones(3000)], s)
    assert eer == pytest.approx(0.25, abs=0.005) and t == pytest.approx(0.75, abs=0.01)


def test_model_shapes():
    x = torch.randn(2, N) * 0.1
    assert MelCNN().eval()(x).shape == (2,)
    st = Student().eval()
    assert st(x).shape == (2,)
    assert n_params(st) < 400_000, "student must stay phone-sized"
    assert W2VHead(25, 2048)(torch.randn(3, 25, 2048)).shape == (3,)


def test_gmm_separates_simple_classes():
    real = np.concatenate([mfcc_frames(tone(1) + rng.standard_normal(SR).astype(np.float32) * 0.01)
                           for _ in range(3)])
    fake = np.concatenate([mfcc_frames(rng.standard_normal(SR).astype(np.float32) * 0.1) for _ in range(3)])
    g = GMMClassifier(4).fit(real, fake)
    assert g.score(rng.standard_normal(SR).astype(np.float32) * 0.1) > g.score(tone(1))


def test_hindi_folder_layout(tmp_path):
    for sub, spk in (("real", "s1"), ("fake", "s1")):
        (tmp_path / sub / spk).mkdir(parents=True)
        sf.write(tmp_path / sub / spk / "a.wav", tone(0.5), SR)
    items = hindi_items(tmp_path)
    assert [(it.label, it.speaker) for it in items] == [(0, "s1"), (1, "s1")]


def test_onnx_export_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    net = Student().eval()
    x = torch.randn(3, N) * 0.1
    path = tmp_path / "s.onnx"
    kw = dict(input_names=["audio"], output_names=["prob"], opset_version=17,
              dynamic_axes={"audio": {0: "batch"}, "prob": {0: "batch"}})
    try:
        torch.onnx.export(net, torch.zeros(1, N), path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, torch.zeros(1, N), path, **kw)
    out = ort.InferenceSession(str(path)).run(None, {"audio": x.numpy()})[0]
    with torch.no_grad():
        ref = net(x).numpy()
    assert np.abs(out - ref).max() < 1e-3
    assert path.stat().st_size < 1.5e6


def test_real_world_augmentations():
    from dfad.augment import OPUS_OK, opus_roundtrip, room_mic, speakerphone
    x = fix_length(tone(1), N)
    y = room_mic(x, rng)
    assert y.shape == x.shape and not np.allclose(x, y)
    s = speakerphone(x, rng)
    assert s.shape == x.shape and s.dtype == np.float32 and np.isfinite(s).all()
    if OPUS_OK:
        z = opus_roundtrip(x, rng)
        assert z.shape == x.shape and not np.allclose(x, z)


def test_speaker_split_is_disjoint():
    from dfad.data import Item, speaker_split
    items = [Item(f"{s}_{i}.wav", i % 2, f"spk{s}") for s in range(20) for i in range(4)]
    parts = {p: {it.speaker for it in speaker_split(items, p)} for p in ("train", "dev", "test")}
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["dev"]) and not (parts["dev"] & parts["test"])
    assert sum(len(v) for v in parts.values()) == 20
