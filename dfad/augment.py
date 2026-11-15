"""Real-world degradations: room/mic, noise, MP3 / Opus compression, telephone channel, gain."""
import io
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from .data import SR, fix_length, load_audio, resample

MP3_OK = "MP3" in sf.available_formats()
OPUS_OK = "OPUS" in sf.available_subtypes("OGG")
_PHONE_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def add_noise(x, rng, snr_db, noises=None):
    if noises:
        n = fix_length(noises[rng.integers(len(noises))], len(x), rng)
    else:  # white or pink-ish noise
        n = rng.standard_normal(len(x)).astype(np.float32)
        if rng.random() < 0.5:
            n = np.cumsum(n) * 0.02
            n -= n.mean()
    px, pn = np.mean(x ** 2) + 1e-10, np.mean(n ** 2) + 1e-10
    snr = rng.uniform(*snr_db)
    return x + n * np.sqrt(px / (pn * 10 ** (snr / 10)))


def mp3_roundtrip(x, rng):
    if not MP3_OK:
        return x
    buf = io.BytesIO()
    try:  # newer soundfile exposes compression_level for MP3 (0 = best, 1 = worst)
        sf.write(buf, x, SR, format="MP3", compression_level=float(rng.uniform(0.3, 0.95)))
    except TypeError:
        buf = io.BytesIO()
        sf.write(buf, x, SR, format="MP3")
    buf.seek(0)
    y = sf.read(buf, dtype="float32")[0]
    if y.ndim > 1:
        y = y.mean(axis=1)
    return fix_length(y, len(x))  # MP3 adds encoder delay / padding


def opus_roundtrip(x, rng):
    """WhatsApp / Telegram voice notes and VoIP calls use Opus."""
    if not OPUS_OK:
        return x
    buf = io.BytesIO()
    try:
        sf.write(buf, x, SR, format="OGG", subtype="OPUS", compression_level=float(rng.uniform(0.3, 0.95)))
    except TypeError:
        buf = io.BytesIO()
        sf.write(buf, x, SR, format="OGG", subtype="OPUS")
    buf.seek(0)
    y = sf.read(buf, dtype="float32")[0]
    if y.ndim > 1:
        y = y.mean(axis=1)
    return fix_length(y, len(x))


def room_mic(x, rng):
    """Small-room echo (synthetic exponentially-decaying impulse response) + cheap-mic tilt EQ."""
    rt60 = rng.uniform(0.15, 0.6)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / rng.uniform(0.3, 1.0)            # direct path vs. reverb balance
    y = fftconvolve(x, rir)[:len(x)]
    lo, hi = rng.uniform(80, 250), rng.uniform(4000, 7500)
    y = sosfilt(butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos"), y)
    return (y * np.sqrt((np.mean(x ** 2) + 1e-10) / (np.mean(y ** 2) + 1e-10))).astype(np.float32)


def phone_channel(x, rng):
    y = resample(x, SR, 8000)
    y = sosfilt(_PHONE_SOS, y)
    mu = 255.0  # G.711-style mu-law 8-bit quantisation
    peak = np.abs(y).max() + 1e-9
    c = np.sign(y / peak) * np.log1p(mu * np.abs(y / peak)) / np.log1p(mu)
    c = np.round(c * 127) / 127
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu * peak
    return fix_length(resample(y.astype(np.float32), 8000, SR), len(x))


def loudspeaker(x, rng):
    """A phone's small loudspeaker: little bass below a few hundred Hz, mild soft clipping."""
    y = sosfilt(butter(2, rng.uniform(250, 600), btype="highpass", fs=SR, output="sos"), x)
    drive = rng.uniform(1.0, 4.0)
    peak = np.abs(y).max() + 1e-9
    return (np.tanh(drive * y / peak) / np.tanh(drive) * peak).astype(np.float32)


def speakerphone(x, rng, snr_db=(10, 25)):
    """The in-call path Call Guard hears: call channel (cellular 8 kHz or VoIP Opus) -> the
    phone's loudspeaker -> room -> the same phone's mic, plus background noise. Training on
    the effects one at a time almost never produces this whole chain."""
    x = phone_channel(x, rng) if rng.random() < 0.5 else opus_roundtrip(x, rng)
    return add_noise(room_mic(loudspeaker(x, rng), rng), rng, snr_db).astype(np.float32)


class Augmenter:
    def __init__(self, cfg, noise_dir=None):
        self.cfg = cfg
        self.noises = []
        if noise_dir:
            files = sorted(Path(noise_dir).rglob("*.wav"))[:500]
            self.noises = [load_audio(f) for f in files]
        if not MP3_OK:
            warnings.warn("libsndfile has no MP3 support; MP3 augmentation disabled "
                          "(pip install -U soundfile)")

    def __call__(self, x, rng):
        c = self.cfg
        if rng.random() > c["p"]:
            return x
        sp = c.get("speakerphone", {"p": 0})
        if rng.random() < sp["p"]:  # the whole in-call chain instead of the separate effects below
            x = speakerphone(x, rng, sp.get("snr_db", (10, 25)))
            if rng.random() < c["gain"]["p"]:
                x = x * 10 ** (rng.uniform(*c["gain"]["db"]) / 20)
            return np.clip(x, -1, 1).astype(np.float32)
        if rng.random() < c.get("room", {"p": 0})["p"]:
            x = room_mic(x, rng)
        if rng.random() < c["noise"]["p"]:
            x = add_noise(x, rng, c["noise"]["snr_db"], self.noises)
        if rng.random() < c["phone"]["p"]:
            x = phone_channel(x, rng)
        r = rng.random()  # at most one lossy codec, like a real recording path
        if r < c["mp3"]["p"]:
            x = mp3_roundtrip(x, rng)
        elif r < c["mp3"]["p"] + c.get("opus", {"p": 0})["p"]:
            x = opus_roundtrip(x, rng)
        if rng.random() < c["gain"]["p"]:
            x = x * 10 ** (rng.uniform(*c["gain"]["db"]) / 20)
        return np.clip(x, -1, 1).astype(np.float32)
