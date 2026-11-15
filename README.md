# Deepfake Audio Detection + Call Guard

Classifies speech as **REAL** or **FAKE** (AI-generated), compares three models by EER, and
brings a tiny version of the model to the phone:

| Part | What it is | Where it runs |
|---|---|---|
| `dfad/`, `scripts/` | data, augmentation, GMM / CNN / XLS-R models, training, EER evaluation | laptop GPU |
| `app.py` | Streamlit demo: upload/record → verdict, per-window chart, suspicion heatmap | laptop |
| `web/` | phone web app (PWA): **Check a clip** + **Call Guard live** (2nd device) | any phone browser, offline |
| `android/` | Android app: checks the caller **on the same phone during a call** (if the phone allows it), after-call recording check, Share-to-check | Android 10+ |

Label convention everywhere: `0 = real (bona fide)`, `1 = fake (spoof)`; scores: higher = more fake.

---

## 1. Setup (Windows, Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build for the RTX GPU
pip install -r requirements.txt
pytest tests -q
```

Edit the dataset paths at the top of **`config.yaml`**:

* `asvspoof_la`: the folder containing `ASVspoof2019_LA_train/ …_dev/ …_eval/` and `ASVspoof2019_LA_cm_protocols/`
* `in_the_wild`: the folder containing `meta.csv` and the wav files
* `deepvoice`: DEEP-VOICE (RVC voice conversion), with `meta.csv` and `audio/`
* `hindi`: `data/hindi/real/<speaker>/*.wav` and `data/hindi/fake/<speaker>/*.wav`

No ffmpeg is needed. `soundfile` handles MP3 augmentation.

## 2. Train and evaluate

The GPU has 4 GB, so XLS-R-300M runs **frozen**. Its pooled layer features are computed once and
cached, and only a small head is trained (it takes minutes). Training data gets noise, MP3,
telephone-channel and gain augmentation.

```powershell
# a) baseline and CNN
python scripts/train.py --model gmm
python scripts/train.py --model cnn

# b) XLS-R: cache features once (fp16, resumable; about 16 GB of disk), then train the head.
#    These are the sets named in config.yaml (w2v/student train_sets + dev_sets) plus the test sets.
python scripts/cache_xlsr.py asvspoof_train itw_train deepvoice_train asvspoof_dev itw_dev deepvoice_dev
python scripts/cache_xlsr.py asvspoof_eval itw_test deepvoice_test
python scripts/train.py --model w2v

# c) the phone model: distil the XLS-R teacher into the ~0.2M-param Student
python scripts/distill.py
python scripts/export_onnx.py --int8     # -> web/model/ and android/app/src/main/assets/

# d) EER table: ASVspoof eval / In-the-Wild test speakers / DEEP-VOICE held-out people / Hindi
python scripts/evaluate.py               # -> results/table.csv, results/table.md
```

Add `--limit 500` (and `--epochs 2`) to any step for a quick smoke run. For caching, pass
`--limit` to `cache_xlsr.py`. Every checkpoint stores its decision threshold, set at the
dev-set EER.

**Hindi test set:** record volunteers (with consent) into `data/hindi/real/<speaker>/`, put
sentences in `data/hindi/sentences.txt`, then run `python scripts/make_hindi_fakes.py`. This
uses Coqui XTTS-v2 and is heavy, so run it on Colab. Then run
`python scripts/cache_xlsr.py hindi` and `python scripts/evaluate.py --sets hindi`.

## 3. Streamlit demo (laptop)

```powershell
streamlit run app.py
```

## 4. Phone web app (`web/`)

It's plain HTML/JS with no build step. The model runs with ONNX Runtime Web (WASM) on the device.
The first load is about 3–4 MB. After that it works offline and can be installed to the home screen.

* **On a laptop:** run `python -m http.server 8000 -d web`, then open http://localhost:8000
* **On a phone:** browsers only allow the microphone over HTTPS. Two options:
  * Push `web/` to GitHub Pages (Settings → Pages) and open the `https://…github.io/…` URL.
  * Or plug the phone into the laptop over USB and run `adb reverse tcp:8000 tcp:8000`. Then
    open `http://localhost:8000` on the phone.

**Check a clip** works on the same phone: a voice note, a call recording, or an 8 s recording.
**Call Guard live** is for a 2nd device placed next to the speakerphone. It scores the last 4 s
every 2 s and skips silence. After 3 red windows in a row it vibrates (or flashes and beeps on
iPhone) and shows the safety tips.

## 5. Android app (`android/`): checking on the same phone during a call

Browsers get silence while a call is active. Android does let an **accessibility service** keep
using the microphone during a call, which is the method call-recorder apps use. The app uses
that, **if the phone allows it**.

* **Modes** (chosen in the app):
  * **Off**
  * **On request:** a "Check caller" bubble appears during calls, tap it to start.
  * **Automatic:** checking starts when a call is answered.

  Both cellular and WhatsApp/VoIP calls are detected. The call must be on **speaker**.
* **Is it allowed?** If the mic delivers pure digital silence, the app says live checking is
  blocked on this phone and offers the fallbacks below.
* **Fallbacks on the same phone:**
  * Choose the dialer's call-recordings folder. After each call the app offers
    "Check this call's recording?".
  * Or **Share** any voice note to Call Guard.
* **Privacy:** there is no INTERNET permission. Audio lives only in a 4 s RAM buffer. Checking is off by default.

Build:
1. Install **Android Studio** (it includes the JDK and SDK).
2. On the phone, enable Developer options → USB debugging.
3. Open the `android/` folder in Android Studio, let Gradle sync, then press Run.
4. For a sideload APK: Build → Build APK(s).

**Milestone A: does in-call listening work on *your* phone?** Build the app *before* training
anything. Without `student.onnx` in `assets/`, it runs in **diagnostic mode**: the bubble
shows the live mic level instead of a verdict.
1. Turn on the Call Guard service and set the mode to On request.
2. Make a call, put it on speaker and tap the bubble.
3. Read the bubble:
   * **"Mic OK −30 dB" that changes while the caller talks:** in-call checking works on this phone.
   * **"Mic BLOCKED":** this phone mutes other apps during calls, so use the fallbacks.

After `export_onnx.py` has copied the model into `assets/`, rebuild. The bubble then shows
green, yellow or red with the AI-voice risk (the decision threshold shows as 50%). Timing is logged by `adb logcat -s CallGuard`.

## Limitations (please state these in the report)

* **iPhone:** Apple gives no app any access to call audio. Use Check a clip after the call
  (iOS 18+ can record calls natively), or a 2nd device.
* **Android** in-call capture depends on the phone model and Android version, and needs speakerphone.
* New voice-cloning tools may fool the model until it is retrained. Heavy noise and poor call
  quality reduce accuracy. The Hindi set is small, so its results are indicative.
* The phone Student is smaller than XLS-R and less accurate. Compare their rows in `results/table.md`.
