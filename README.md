# Phoneme recognition (CTC) — Colab training, live testing on a Windows PC

Implements *Phoneme Recognition on PC* (pipeline and network spec). Plan, decisions and
progress: `phoneme-pipeline-plan.md` (also in the Claude project).

## Status
| Part | State |
|---|---|
| Label map (41 classes), text normalisation, CMUdict lexicon, phonemizer | implemented, unit-tested |
| Streaming front end (DC/HPF, min-stats noise, Wiener/bypass, mel40, PCEN, F0, voicing, soft VAD) | implemented, unit-tested incl. chunking-equivalence |
| Greedy CTC decoding, PER (S/D/I + alignment) | implemented, unit-tested |
| Model A / Model B (PyTorch) | implemented; tests need PyTorch (run on your PC / Colab) |
| Recorder (16 kHz mono 16-bit WAV + manifest) | implemented; needs a real mic test on your PC |
| Colab training notebook, evaluation, calibration, live app | next milestones |

## Setup on Windows 11 (PowerShell)

```powershell
cd D:\Usman\phoneme-recognition
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m pytest -q -s
```
Expected: all tests pass (none skipped), and `test_models.py` prints the Model A / B
parameter counts and the measured receptive field (53 frames).

## Recording
Before recording, in Windows Settings → System → Sound → your input device, turn
**Audio enhancements** off (Windows otherwise applies noise suppression / gain control).

```powershell
python scripts/record.py --list-devices
python scripts/record.py --speaker spk01 --session s001 --noise quiet --distance near --mic laptop-internal
```

## Live phoneme recognition (P7)
```powershell
python scripts/live.py --ckpt checkpoints/best.pt --calib reports/model_a_v1_calib/calibration.json --prompt "The tall dog dug a deep pit by the old tree."
```
Say the sentence, then Ctrl+C. Output: `PHONEME(level)` with calibrated confidence 0-7, about 0.3 s behind
speech (Model A needs 260 ms of following audio; it waits, never guesses). Logs go to `logs/live/<time>/`.
`--wav file.wav` runs a recording through the identical live path. Model A only (Model B's BiGRU is not streamable).

## Layout
```
configs/labels.json      frozen 41-class map (blank=0, SIL=1, phonemes 2-40)
src/phonrec/             audio, phonemize, frontend, models, decoding, evaluation, ...
scripts/                 record.py (more to come)
tests/                   pytest suite
data/prompts/            prompt sentences (recordings themselves are git-ignored)
```
