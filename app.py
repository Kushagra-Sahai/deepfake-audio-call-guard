"""Streamlit demo (laptop):  streamlit run app.py"""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dfad import load_config
from dfad.data import SR, load_audio
from dfad.infer import Detector

cfg = load_config()
CKPT = Path(cfg["paths"]["checkpoints"])
LABELS = {"w2v": "wav2vec 2.0 / XLS-R (main)", "cnn": "CNN on mel-spectrogram", "student": "Phone model (tiny)"}

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
st.title("Deepfake Audio Detector")
st.caption("Is this voice a real human or AI-generated? Audio is analysed locally and not stored.")

available = [m for m in LABELS if (CKPT / f"{m}.pt").exists()]
if not available:
    st.error(f"No trained models found in {CKPT}. Train one first (see README).")
    st.stop()
choice = st.selectbox("Model", available, format_func=LABELS.get)


@st.cache_resource(show_spinner="Loading model…")
def get_detector(name):
    return Detector(CKPT / f"{name}.pt")


tab_up, tab_rec = st.tabs(["Upload a file", "Record"])
with tab_up:
    up = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "ogg"])
with tab_rec:
    rec = st.audio_input("Record a voice sample (at least 4 s)")
src = rec or up
if src is None:
    st.info("Upload or record audio to analyse it.")
    st.stop()

try:
    x = load_audio(io.BytesIO(src.getvalue()))
except Exception as e:  # unsupported codec etc.
    st.error(f"Could not read this audio ({e}). Try WAV, FLAC, MP3 or OGG.")
    st.stop()
if len(x) < SR:
    st.warning("Very short clip (< 1 s): the result will be unreliable.")

det = get_detector(choice)
with st.spinner("Analysing…"):
    res = det.analyse(x)
fake = res["label"] == "FAKE"

c1, c2 = st.columns(2)
c1.metric("Verdict", "AI-generated (FAKE)" if fake else "Likely REAL", delta=None)
c2.metric("AI-voice risk", f"{res['risk'] * 100:.0f}%", help="The decision threshold is 50%")
if fake:
    st.error("This voice shows signs of AI generation. Verify through another channel before "
             "sending money or sharing an OTP.")
else:
    st.success("No strong signs of AI generation were found. This is not a guarantee.")

st.subheader("Fake probability over time")
fig, ax = plt.subplots(figsize=(7, 2.2))
t = res["window_starts"]
ax.bar(t, res["window_probs"], width=det.hop / SR * 0.9, align="edge",
       color=["#d9534f" if q >= det.threshold else "#5cb85c" for q in res["window_probs"]])
ax.axhline(det.threshold, ls="--", c="k", lw=1, label="threshold")
ax.set(ylim=(0, 1), xlabel="window start (s)", ylabel="P(fake)")
ax.legend(loc="upper right", fontsize=8)
st.pyplot(fig, clear_figure=True)

st.subheader("Where the model looked")
fast = choice != "w2v" or det.device == "cuda"
with st.spinner("Computing suspicion heatmap…"):
    heat, w = det.heatmap(x, *((16, 8) if fast else (8, 4)))
import librosa  # noqa: E402

S = librosa.power_to_db(librosa.feature.melspectrogram(y=w, sr=SR, n_fft=512, hop_length=128, n_mels=64))
fig, ax = plt.subplots(figsize=(7, 3))
dur = len(w) / SR
ax.imshow(S, origin="lower", aspect="auto", cmap="gray", extent=(0, dur, 0, SR / 2))
ax.imshow(heat, origin="lower", aspect="auto", cmap="inferno", alpha=0.45, extent=(0, dur, 0, SR / 2),
          interpolation="bilinear")
ax.set(xlabel="time (s)", ylabel="frequency (Hz)", title="Bright = removing this region lowers P(fake)")
st.pyplot(fig, clear_figure=True)
st.caption(f"Model: {LABELS[choice]} · decision threshold {det.threshold:.2f} (set at the dev-set EER)")
