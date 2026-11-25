# Phoneme recognition (CTC): train in Colab, recognise live on a Windows PC

Speech in, English phonemes out, each with a confidence from 0 to 7. Implements the spec
*Phoneme Recognition on PC* (pipeline and network spec). Later target (out of scope here): the Talk-E
robot (INMP441 mic → ESP32-C6, Model A quantised to INT8 for the Luckfox Pico Zero).

Plan, assumptions, decisions (D1–D12) and the dated progress log with every measured number:
[`phoneme-pipeline-plan.md`](phoneme-pipeline-plan.md).

## How it works

```
 Windows PC                         Google Colab (T4 GPU)                     Windows PC
 ──────────                         ─────────────────────                     ──────────
 record.py: mic → 16 kHz WAV   ┐    LibriSpeech + transcripts                 live.py: mic (44.1 kHz)
 + transcript + manifest.csv   ├──▶ → phonemes (CMUdict, 39 + SIL + blank)    → same front end, streaming
                               │    → front end → 42-ch features (cached)     → Model A (waits 260 ms
 make_code_zip.py → Drive  ────┘    → train Model A / B with CTC                for following audio)
                                    → dev evaluation + log-probs   ──Drive──▶ → greedy CTC → phoneme
                                                                              + confidence 0–7 + logs
```

**Front end** (`src/phonrec/frontend/streaming.py`). The *same* class makes the training features and the
live features, so they are identical by construction (a test checks this):
DC blocker + 80 Hz high-pass → 25 ms Hann window, 10 ms hop, 512-pt FFT → minimum-statistics noise floor →
Wiener filter (or bypass) → 40 mel bands 80–7600 Hz → PCEN → plus log-F0 and voicing = **42 channels**.
Every stage is causal. A soft VAD probability is a side output (used live for line breaks only).

**Labels** (`configs/labels.json`, frozen): blank = 0, SIL = 1, 39 ARPAbet phonemes = 2–40. CMUdict with
stress removed; LibriSpeech lexicon and `lexicon/overrides.txt` for other words; utterances with unknown
words are dropped and listed in `phonemize_report.json`.

**Models** (input 42 × T at 10 ms, output 41 log-probs at 20 ms, CTC loss):

| | Model A (live, later on the robot) | Model B (PC only, stronger) |
|---|---|---|
| Layers | Conv k5 s2 42→128, 4 depthwise-separable blocks (dil 1,1,2,2), 1×1 head | Conv k5 s2 42→256, 4 DS blocks, 2-layer BiGRU 256, linear head |
| Parameters (measured) | 134,569 | 2,318,889 |
| Context | 53 frames = 530 ms, of which 26 frames (260 ms) in the future | whole utterance (BiGRU) |
| Live streaming | yes, exact | no (needs future audio) |

**Decoding and confidence.** Greedy CTC (argmax per frame, merge repeats, drop blanks). A phoneme's
confidence is its highest posterior over its frames, after a temperature fitted on the **dev** set
(`scripts/calibrate.py`). Level = floor(8 × confidence), so level 6 claims "right 75–87.5 % of the time".

## Results so far (all measured; details and dates in the plan log)

First run: Colab, `QUICK=True` = 5,678 LibriSpeech train-clean-100 utterances (~20 % of it), 40 epochs.
Dev = LibriSpeech dev-clean, 2,523 utterances. Test-clean has **not** been run (kept for the very end).

| | Model A | Model B |
|---|---|---|
| Dev PER (greedy, SIL excluded) | 0.4245 | 0.2463 |
| Training time per epoch (T4) | ~40 s | ~64 s |
| Dev calibration (A) | T = 0.995; held-out ECE 0.0158 → 0.0150; every 0–7 level within its range except level 1 at its edge (0.251) | not done yet |

Live on the laptop mic (Model A, one sentence, 3 tries, 2026-11-11): latency median 0.30 s (p95 ≤ 0.46 s),
0 dropped audio blocks, processing 0.11 × real time; **PER 0.75–0.93**. On this mic the confidence levels
do not separate right from wrong phonemes yet. Both problems are expected: the model has never heard this
mic or room. Fixing that is milestone P9 (own recordings + fine-tuning + re-calibration on an own-mic dev set).

| Milestone | State |
|---|---|
| P0–P3 labels, recorder, front end, models + tests | done |
| P4 Colab training of Model A | done for the QUICK subset; full train-clean-100 run still to do |
| P5 dev evaluation | done for A |
| P6 temperature + 0–7 confidence | done for A (LibriSpeech dev) |
| P7 live app | done |
| P8 Model B | trained; dev eval / calibration not run |
| P9 own recordings, fine-tune, Wiener A/B | **next**: needs recordings; split + fine-tune tooling not written yet |
| P10 distillation, beam search + LM | optional, later |

## How to run it

### 1. Setup on the PC (once)
Windows 11, PowerShell, Python 3.13:
```powershell
cd D:\Usman\phoneme-recognition
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m pytest -q -m "not slow"     # 76 tests should pass
```
**Always activate the venv** (the prompt shows `(.venv)`). Without it, Windows' own Python runs and
fails with errors such as `No module named 'cmudict'`.
Optional end-to-end check on synthetic data (its PER means nothing): `python scripts/smoke_test.py`.

### 2. Send the code to Colab
```powershell
python scripts/make_code_zip.py       # -> phonrec_code.zip (code only, no audio or checkpoints)
```
Upload it to **My Drive/phonrec/phonrec_code.zip** (exact name; replace the old one). Repeat after
every code change.

### 3. Train in Colab
Open `notebooks/train_colab.ipynb`, Runtime → Change runtime type → **T4 GPU**, run in order:

| Section | Does | Notes |
|---|---|---|
| 1 Setup | mounts Drive, unzips code, runs tests | expect `59 passed` or more, `cuda True` |
| 2 Data | downloads LibriSpeech (~7 GB, once) to Drive, extracts locally | re-run resumes |
| 3 Features | phonemizes, builds 42-ch features on Drive | `QUICK=True` = 6,000 utts; `False` = all |
| 3b Report | prints kept/dropped utterances and unknown words | |
| 4 Copy | features → local disk for fast training | |
| 5 Train A | checkpoints to `My Drive/phonrec/runs/model_a_v1/` every epoch | re-run resumes after a disconnect |
| 6 Curves | loss and dev PER plots | |
| 7 Dev eval | PER, S/D/I, confusions, latency, `logprobs.npz` | needed for calibration |
| 8 Model B | same for Model B | |
| 9 Test | **once, at the very end** | never tune anything on it |

For a new experiment, use a new run folder (e.g. `model_a_v2`); re-running into a finished folder only resumes it.

### 4. Bring the results to the PC
| From `My Drive/phonrec/` | To `D:\Usman\phoneme-recognition\` |
|---|---|
| `runs/model_a_v1/best.pt` | `checkpoints\best.pt` |
| `runs/model_b_v1/best.pt` | `checkpoints\model_b_best.pt` |
| `reports/model_a_v1_dev/` (`results.json`, `logprobs.npz`, `per_utterance.jsonl`) | `reports\model_a_v1_dev\` |

### 5. Calibrate the confidence (dev only)
```powershell
python scripts/calibrate.py --logprobs reports/model_a_v1_dev/logprobs.npz --per-utt reports/model_a_v1_dev/per_utterance.jsonl --out reports/model_a_v1_calib
```
Writes `calibration.json` (temperature, level table) and `reliability.png`. The script refuses test-split files.

### 6. Live recognition
```powershell
python scripts/live.py --ckpt checkpoints/best.pt --calib reports/model_a_v1_calib/calibration.json --prompt "The tall dog dug a deep pit by the old tree."
```
Wait for `listening...`, speak, press **Ctrl+C**. Phonemes appear as `PHONEME(level)` about 0.3 s after
you say them; a new line starts after each pause. With `--prompt` it prints REF, HYP and PER at the end.
- `--wav file.wav` runs a recording through the identical live path (gives exactly the offline result).
- `--list-devices`, `--device N` pick the microphone; `--seconds N` stops automatically.
- Each run is logged in `logs/live/<time>/`: `audio16k.wav`, `tokens.jsonl`, `summary.json`
  (latency, overflows, PER and the individual errors).

### 7. Record your own data (for P9)
Before recording: Windows Settings → System → Sound → input device → turn **Audio enhancements** off.
```powershell
python scripts/record.py --list-devices
python scripts/record.py --speaker spk01 --session s001 --noise quiet --distance near --mic laptop-internal
```
Per prompt (`data/prompts/prompts_v1.txt`, 44 sentences): Enter to start, Enter to stop, then keep / redo /
skip. Saves 16 kHz mono 16-bit WAVs and a transcript per file under `data/recordings/<session>/` and a row
in `data/manifest.csv`. Keep the distance and setup fixed within a session; record quiet and noisy
sessions; keep whole sessions apart as a test set that is never trained on.

## Project layout
```
configs/labels.json            frozen 41-class label map
configs/train_a.json, train_b.json   training settings (lr, batch, epochs, SpecAugment, ...)
lexicon/overrides.txt          manual pronunciations (win over CMUdict)
data/prompts/prompts_v1.txt    sentences to record
data/manifest.csv              one row per own recording (audio itself is git-ignored)
notebooks/train_colab.ipynb    the Colab workflow above
src/phonrec/
  audio/io.py                  WAV read/write, resampling, level check
  phonemize/                   text normalisation, lexicon, transcript -> phonemes
  frontend/                    streaming front end (filters, noise, Wiener, mel, PCEN, pitch, VAD)
  data/                        LibriSpeech manifests, noise augmentation, sharded feature store
  models/                      Model A, Model B, building blocks
  training/                    dataset, bucketing, SpecAugment, CTC training loop, checkpoints
  decoding/greedy.py           greedy CTC decoding
  evaluation/                  PER (S/D/I), evaluation report, calibration (temperature, 0-7)
  streaming/live.py            exact streaming: resampler, Model A window, greedy decoder
scripts/
  record.py                    recorder
  prepare_librispeech.py       LibriSpeech -> manifests
  build_features.py            manifest -> phonemes + cached features
  make_code_zip.py             code zip for Colab
  smoke_test.py                end-to-end test on synthetic data
  calibrate.py                 temperature + 0-7 table from dev log-probs
  live.py                      live recogniser
tests/                         pytest suite (76 fast tests + 1 slow smoke test)
phoneme-pipeline-plan.md       requirements, decisions, milestones, progress log
```
Git-ignored (stay on the PC / Drive): `.venv/`, audio, features, `checkpoints/`, `reports/`, `logs/`,
`smoke_out/`, `phonrec_code.zip`.

## Rules the project keeps
- Training and live use the same front end; the live path never uses future audio.
- Checkpoint choice, calibration and the temperature use the dev split only; the test split is run once, at the end.
- `configs/labels.json` is never changed.
- Only measured numbers are reported; assumptions are logged as decisions in the plan.

## Known issues
- The LibriSpeech lexicon copy on Drive may be truncated (common LibriSpeech words such as BOZZLE stayed
  unknown, ~5–6 % of utterances dropped). Check in Colab: `!wc -l` on
  `My Drive/phonrec/datasets/librispeech-lexicon.txt` (a complete file has ~200,000 lines).
- Model A has only been trained on 20 % of train-clean-100; the full run is still to do.
- Confidence is calibrated for LibriSpeech audio only; on the laptop mic it is not yet meaningful.

## Data and licences
LibriSpeech (OpenSLR 12) is CC BY 4.0: attribute it in reports. CMUdict via the `cmudict` package.
Check the OpenSLR 11 page for the LibriSpeech lexicon's terms.
