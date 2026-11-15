# App simulation: `student.onnx` (threshold 0.9386)

Accuracy = balanced (mean of real-correct and fake-correct). Real flagged = false alarms on genuine voices. EER is threshold-free (how separable the scores are).

## In-the-Wild (new speakers)

**Check a clip** (300 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 90.7% | 11.3% | 92.7% | 8.7% |
| voice note (Opus) | 90.8% | 11.3% | 93.0% | 8.7% |
| phone call (8 kHz) | 87.7% | 14.3% | 89.7% | 12.0% |
| speakerphone | 61.5% | 13.7% | 36.7% | 35.7% |
| noisy room (5-10 dB) | 87.8% | 12.7% | 88.3% | 11.7% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 13% | 20 s | 16% |
| AI voice | 30 | 67% | 14 s | 46% |

## DEEP-VOICE RVC (new people)

**Check a clip** (203 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 73.1% | 1.5% | 47.7% | 16.0% |
| voice note (Opus) | 73.8% | 2.0% | 49.7% | 16.3% |
| phone call (8 kHz) | 67.6% | 2.5% | 37.7% | 19.7% |
| speakerphone | 63.1% | 4.4% | 30.7% | 24.0% |
| noisy room (5-10 dB) | 68.3% | 2.0% | 38.7% | 21.2% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 30% | 26 s | 20% |
| AI voice | 30 | 80% | 13 s | 45% |

## ASVspoof eval (new TTS attacks)

**Check a clip** (300 real / 300 fake clips)

| how the audio reached the phone | accuracy | real flagged as fake | fakes caught | EER |
|---|---|---|---|---|
| clean | 90.5% | 2.0% | 83.0% | 7.0% |
| voice note (Opus) | 90.5% | 2.3% | 83.3% | 7.0% |
| phone call (8 kHz) | 87.8% | 5.3% | 81.0% | 11.3% |
| speakerphone | 63.0% | 25.7% | 51.7% | 34.7% |
| noisy room (5-10 dB) | 78.2% | 8.0% | 64.3% | 18.3% |

**Live call on speakerphone** (~45 s calls, alert after 3 red windows in a row)

| caller | calls | alert raised | median time to alert | red windows |
|---|---|---|---|---|
| real person | 30 | 30% | 14 s | 25% |
| AI voice | 30 | 83% | 20 s | 49% |

