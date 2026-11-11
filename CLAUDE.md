# CLAUDE.md — phoneme-recognition

CSE student project: CTC phoneme recogniser. Record on Windows PC → train in Google Colab → live test on PC.
Read `phoneme-pipeline-plan.md` first (requirements, assumptions, decisions D1–D10, milestones P0–P10, progress log).

## Sources
- PRIMARY spec: "Phoneme Recognition on PC" (pc_phoneme_pipeline.pdf). Follow it exactly.
- Background only: "Talk-E ML Engineering Spec" — later target robot: INMP441 mic → ESP32-C6; Model A later
  INT8-quantised for the Luckfox Pico Zero. Out of scope for the PC phase.
- IGNORE: "Talk-e Vision Pipeline Brief" (a different assignment).

## Hard rules
- Never change `configs/labels.json` (blank=0, SIL=1, phonemes 2–40).
- Training and live inference use the same front end (`src/phonrec/frontend/streaming.py`).
  The live path never uses future audio (it waits for the ~26-frame lookahead instead).
- Never tune anything on the test split. Checkpoint choice, calibration and temperature use dev only.
  The test split is run once, at the end.
- Never report numbers that were not actually measured. Say clearly what is untested.
- If the spec is ambiguous: state the assumption and log it as a decision (D-number) in the plan.
- After each milestone: update `phoneme-pipeline-plan.md` (checklist + progress log).
- Git: repo github.com/nhsabz/phoneme-recognition (branch main). No `Co-Authored-By: Claude` or other
  Claude attribution in commit messages or PR descriptions.

## Environment
- Windows 11, Python 3.13, venv `.\.venv` (always use it: `.\.venv\Scripts\Activate.ps1` or `.venv/Scripts/python.exe`).
- Mic: laptop built-in "Microphone Array (AMD Audio Dev)", 44.1 kHz → resampled to 16 kHz.
- Training: Google Colab T4 GPU (TPU not used). Drive 400 GB. Persistent dir: `My Drive/phonrec/`.
- Code goes to Colab as `My Drive/phonrec/phonrec_code.zip` (built by `python scripts/make_code_zip.py`).

## Commands
- Tests: `python -m pytest -q -m "not slow"` (76 pass on this PC as of 2026-11-10)
- Smoke test (synthetic data, end to end): `python scripts/smoke_test.py`
- Zip for Colab: `python scripts/make_code_zip.py`
- Calibrate (dev only): `python scripts/calibrate.py --logprobs reports/model_a_v1_dev/logprobs.npz --per-utt reports/model_a_v1_dev/per_utterance.jsonl --out reports/model_a_v1_calib`
- Live: `python scripts/live.py --ckpt checkpoints/best.pt --calib reports/model_a_v1_calib/calibration.json [--wav f.wav] [--prompt "..."]`
- Notebook: `notebooks/train_colab.ipynb` (sections: 1 setup, 2 data, 3 features, 4 copy, 5 train A,
  6 curves, 7 dev eval, 8 Model B, 9 final test once)

## Status (2026-11-02)
- Measured on this PC: Model A 134,569 params, Model B 2,318,889; receptive field 53 frames (26 future).
  Front end RTF ≈ 0.010 per core (15 s noise clip).
- `scripts/smoke_test.py` PASSED (features → train A, loss 15.1 → 5.1 → resume → evaluate → train B).
  Synthetic data, so its PER means nothing.
- Recorder works; only 1 test recording so far. Own recordings come later (P9).
- 2026-11-10: first real Model A run (Colab, QUICK=True, ~5.7k LibriSpeech utts, 40 epochs) →
  `checkpoints/best.pt`, best dev-clean PER 0.4245 at the last epoch (not converged).
  One own-mic utterance decoded at PER 0.679 (single sample, not a metric).
- P5 dev eval done (A: PER 0.4246); Model B trained (dev PER 0.2463). P6 done for A: T = 0.995 (dev).
- P7 live app written (`streaming/live.py`, `scripts/live.py`); WAV mode verified identical to offline.
  Mic runs 2026-11-11: latency median 0.30 s, 0 overflows, RTF ~0.11; PER 0.75-0.93 on one sentence (3 tries);
  confidence levels do not separate right/wrong on this mic (LibriSpeech calibration does not transfer).
- Files on PC (git-ignored): checkpoints/best.pt (A), checkpoints/model_b_best.pt (B), reports/.
- Next: P9 own recordings / fine-tune (+ own-mic dev set for re-calibration). Later: QUICK=False full train-clean-100 run;
  check Drive librispeech-lexicon.txt is complete (suspected truncated, see plan log).
