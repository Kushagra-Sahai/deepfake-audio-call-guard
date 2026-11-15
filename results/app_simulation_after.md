# App simulation: `student.onnx` (threshold 0.8931, live-call threshold 0.9736)

Accuracy = balanced (mean of real-correct and fake-correct). Real flagged = false alarms on genuine voices. EER is threshold-free (how separable the scores are).

## In-the-Wild (new speakers)

**Check a clip** (300 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 92.8% | 11.0% | 96.7% | 5.3% |
| voice note (Opus) | 93.0% | 10.7% | 96.7% | 5.7% |
| phone call (8 kHz) | 89.3% | 16.3% | 95.0% | 10.3% |
| speakerphone | 72.5% | 18.7% | 63.7% | 28.7% |
| noisy room (5-10 dB) | 88.8% | 14.3% | 92.0% | 11.0% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 10% | 30 s | 10% |
| AI voice | 30 | 73% | 10 s | 54% |

## DEEP-VOICE RVC (new people)

**Check a clip** (203 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 73.9% | 2.5% | 50.3% | 14.7% |
| voice note (Opus) | 73.7% | 3.0% | 50.3% | 14.7% |
| phone call (8 kHz) | 74.9% | 1.5% | 51.3% | 14.3% |
| speakerphone | 72.1% | 2.5% | 46.7% | 16.7% |
| noisy room (5-10 dB) | 71.3% | 2.0% | 44.7% | 19.7% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 7% | 26 s | 13% |
| AI voice | 30 | 73% | 12 s | 41% |

## ASVspoof eval (new TTS attacks)

**Check a clip** (300 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 89.8% | 2.7% | 82.3% | 5.7% |
| voice note (Opus) | 90.0% | 3.0% | 83.0% | 6.3% |
| phone call (8 kHz) | 91.3% | 5.3% | 88.0% | 8.7% |
| speakerphone | 79.2% | 25.3% | 83.7% | 20.3% |
| noisy room (5-10 dB) | 82.3% | 7.3% | 72.0% | 16.0% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 10% | 14 s | 14% |
| AI voice | 30 | 80% | 21 s | 45% |

