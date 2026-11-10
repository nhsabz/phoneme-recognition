# Phoneme Recognition on PC — Plan, Decisions & Progress

Source spec: *Phoneme Recognition on PC — pipeline and network spec* (4 pages). Workflow: record on Windows PC → train in Google Colab → live test on PC with USB mic.
Status legend: [ ] not started · [~] in progress · [x] done with evidence · (A) = assumption where spec is silent · (D) = deliberate deviation, quantified.

---

## 1. Requirements checklist (from the spec)

### Data
- [~] Recorder: Python `sounddevice`, 16 kHz, mono, 16-bit PCM WAV, one transcript per file
- [ ] Quiet + noisy rooms; same mic and distance rules per session; several speakers if possible
- [ ] Held-out "never trained on" test set
- [ ] Pretrain on public English corpus with word transcripts (LibriSpeech / Common Voice; check licence), fine-tune on own mic recordings
- [ ] Later: fine-tune on final chain (INMP441 → C6) before anything goes to the Zero (out of scope for PC phase, but noted)

### Labels
- [x] CMUdict → 39 phonemes, stress removed
- [x] OOV words: G2P tool or manual entry
- [x] 41 outputs = 39 phonemes + silence + CTC blank; silence only at utterance edges and long pauses

### Front end (identical for training and live)
- [x] DC removal + high-pass ≈80 Hz
- [x] Minimum-statistics noise floor per FFT bin
- [~] One Wiener filter (decision-directed SNR, min gain floor), always A/B-tested vs bypass; keep/drop decided by PER, not by sound
- [x] No spectral subtraction on top (redundant)
- [x] Soft VAD, ≈200 ms pre-roll; no hard gate
- [x] 25 ms window, 10 ms hop, 512-pt FFT, 40 mel bands 80–7600 Hz
- [x] PCEN (preferred; streaming-capable) or log-mel
- [x] + F0 and voicing strength → 42 channels
- [x] Normalisation: per-utterance CMVN or PCEN's own
- [ ] Dropped: gammatone, formant peak-picking, explicit deltas; MFCC optional

### Models (42 ch @ 10 ms in → 41 log-probs @ 20 ms out, CTC)
- [x] Model A: Conv k5 s2 42→128; 4 DS blocks k5, dil 1,1,2,2, widths 128→192; 1×1 head → 41. ≈131K params, ≈530 ms (53-frame) context
- [x] Model B: Conv k5 s2 42→256; 4 DS blocks k5, dil 1,1,2,2, width 256; 2-layer BiGRU 256; Linear 512→41. ≈2.3M params
- [ ] Spec note: param counts are arithmetic design numbers, not trained results

### Training
- [~] CTC, AdamW, peak LR 1e-3 (A) / 5e-4 (B), WD 0.01, grad clip 5
- [~] Length-bucketed batches 32–64, 30–60 epochs, AMP on Colab GPU
- [~] SpecAugment: 2 freq masks ≤8 bands, 2 time masks ≤40 frames; random gain; added noise
- [~] Checkpoints to Drive (sessions end)

### Evaluation & decoding
- [~] Greedy CTC (argmax, merge repeats, drop blanks) → PER on held-out set
- [ ] PER per condition (quiet/noisy/near/far), confusion matrix, reliability curve
- [x] Temperature fitted on dev set so 0–7 confidence "means what it says" (Model A, LibriSpeech dev; T=0.995)
- [ ] Target PER decided before training and written next to the test set
- [ ] Later: lexicon beam search + LM on same posteriors

### Live test
- [ ] Stream USB mic through identical feature code, short sliding windows, phoneme stream with 0–7 confidence, log failures

---

## 2. Architecture

```
PC: recorder ──WAV+txt+manifest──▶ Drive
Colab: phonemize (CMUdict/G2P) → frontend (DC/HPF → STFT → min-stats noise → [Wiener|bypass] → mel40 → PCEN) + (F0, voicing)
       → 42×T features (cached .npy) → SpecAugment/gain/noise → Model A|B → CTC
       → greedy decode → PER/confusion → temperature on dev → export .pt/.onnx + config + label map
PC: live app: mic callback → ring buffer → same streaming frontend (stateful) → sliding-window model → greedy decode
       → provisional stream + finalised phonemes, 0–7 confidence, latency log
```

One `frontend` package, two entry points: `process_utterance()` (offline) is implemented as a loop over the same `StreamingFrontend.push(chunk)` used live, so training and live features are identical by construction (verified by a test: offline vs chunked output must match to float tolerance).

---

## 3. Verified-by-arithmetic numbers (to be re-verified by code/tests)

Receptive field (both models): stem k5 → 5 frames, jump 2; each DS block adds (5−1)·d·2 = 8d frames; d = 1,1,2,2 → 5 + 48 = **53 input frames = 530 ms** ✔ matches spec.

Model B (no BatchNorm counted): stem 54,016 + 4 DS blocks 269,312 + BiGRU L1 789,504 + BiGRU L2 1,182,720 + head 21,033 = **2,316,585 ≈ 2.3M** ✔ matches spec.

Model A — "widths 128 to 192" is ambiguous. Candidates (no BN / with BN):
| Block widths (in→out) | Params no BN | With BN |
|---|---|---|
| 128→128, 128→128, 128→192, 192→192 | 133,225 | 135,657 |
| 128→128, 128→144, 144→160, 160→192 | 127,481 | 129,849 |
| 128→128, 128→160, 160→192, 192→192 | 143,689 | 146,249 |
**Proposed default (A):** 128→128, 128→128, 128→192, 192→192 → 133,225 (+1.7% vs 131K); allows residuals on 3 of 4 blocks. To confirm.

---

## 4. Ambiguities, assumptions, risks

1. **Model A widths** — see §3. (A) default above.
2. **Causality / lookahead.** The 53-frame context is symmetric if padded "same": each output needs ≈26 frames (≈260 ms) of future audio. Live output is therefore delayed ≈260 ms + window; the live app never uses audio not yet recorded — it waits. Alternative (D): causal padding (no lookahead) — likely worse PER. Decision: train "same" padding, live latency ≈ 300 ms; optionally test causal variant later.
3. **Model B in streaming.** BiGRU needs the whole window; live use re-runs B over a sliding window (e.g. 2–3 s) and only finalises phonemes away from the right edge. B is a PC accuracy/teacher model, not a real-time target.
4. **Per-utterance CMVN is not streamable.** (A) Use PCEN's own normalisation + global feature mean/std from the training set (fixed constants), which is identical offline and live.
5. **F0 / voicing method not specified.** (A) Frame-wise normalised autocorrelation (YIN-style) on the HPF signal over the 25 ms frame (extended to ~40 ms for low F0 if needed, past-only): channel 41 = voicing strength (peak NACF, 0–1); channel 42 = log-F0 normalised, set to 0 when unvoiced. Causal, cheap, portable to C.
6. **Where the Wiener filter acts.** (A) In the STFT power domain feeding the mel bank; no time-domain resynthesis needed for recognition. Bypass flag → gain = 1.
7. **Noise-tracker warm-up.** Minimum statistics needs ~1–1.5 s to settle. (A) Every training utterance is processed from the same cold initial state as the live app, or the live app is warmed for 1.5 s before display. Must be identical in both paths.
8. **Soft VAD role.** "Soft" → (A) VAD produces a speech probability used to (a) drive noise-floor updates, (b) decide when the live app runs/finalises, with 200 ms pre-roll; it does not zero features during training. SIL labels not derived from VAD for LibriSpeech.
9. **Silence labels.** (A) Add SIL at start and end of every utterance. "Long pauses": internal SIL only for own recordings where pauses are marked, or via VAD pause >300 ms (to decide). LibriSpeech: edges only.
10. **Blank vs silence.** Blank = ID 0 (CTC convention in PyTorch), SIL = ID 1, phonemes 2–40 in fixed ARPAbet order. Frozen in `labels.json`, checked by tests.
11. **PCEN parameters** not given. (A) Standard values (s≈0.025, α≈0.98, δ=2, r=0.5, ε=1e-6), fixed, not learned at first.
12. **PER target** not given; spec says decide before training. Needs a decision (yours / supervisor's).
13. **USB mic at 16 kHz on Windows** — many mics run natively at 44.1/48 kHz; WASAPI/MME may resample. (A) Record at native rate and resample to 16 kHz with a fixed high-quality resampler, same in recorder and live app.
14. **Compute.** Free Colab GPU is not guaranteed and sessions are time-limited; feature extraction for 100 h in NumPy will be slow → cache features to Drive, resume-able jobs, start with a subset.
15. **Domain gap.** LibriSpeech (read audiobooks) ≠ your mic/room; and PC mic ≠ INMP441 (spec caution). Fine-tuning on own data is required; final hardware needs another fine-tune.
16. **Little own data.** With 1–2 speakers, a speaker-independent test set is impossible; test set will be session-held-out only, and results will not generalise to new speakers. Reported honestly.

---

## 5. Dataset & training strategy

- **Pretraining:** LibriSpeech `train-clean-100` (~100 h, 251 speakers, CC BY 4.0, OpenSLR 12) — speaker-disjoint official splits; `dev-clean` for validation/temperature, `test-clean` as public test. Licence to re-confirm at download. Later optionally `train-clean-360`/`train-other-500` or Common Voice English (CC0) for accent variety.
- **Noise:** MUSAN (CC BY 4.0) for added-noise augmentation (confirm), plus your own noisy-room recordings.
- **Labels:** CMUdict (BSD-style licence) via `cmudict` package; OOV via `g2p_en` (Apache-2.0) + manual `lexicon_overrides.txt`; OOV report per run. Optional cross-check with LibriSpeech's own lexicon.
- **Fine-tuning:** own recordings, split by **speaker and session** (test sessions recorded separately and frozen); balance quiet/noisy ~50/50 by sampling weights; lower LR (~1/5 of peak), fewer epochs.
- **Overfitting monitor:** train vs dev loss and dev PER per epoch; early stop on dev PER (patience ~8).
- **More data vs model change:** if train PER ≪ dev PER → more data/augmentation; if both high → model/feature change; per-phoneme/per-condition breakdown targets recordings.

---

## 6. Milestones

| ID | Milestone | Depends on | Exit evidence |
|---|---|---|---|
| P0 | Repo skeleton, labels.json, phonemizer + tests | – | pytest green, OOV report |
| P1 | Recorder + manifest + split tool (Windows) | P0 | sample WAVs verified 16k/mono/16-bit |
| P2 | Front end (streaming), 42-ch features, Wiener bypass, VAD + tests | P0 | offline == chunked test, shape/finite tests |
| P3 | Model A/B code; param + receptive-field tests | P0 | printed counts vs spec |
| P4 [~] | Colab notebook: LibriSpeech subset → cache → train A (resume, AMP) | P2,P3 | dev PER curve, best ckpt on Drive |
| P5 [~] | Evaluation: PER, S/D/I, confusion, per condition, latency | P4 | report with real numbers |
| P6 [x] | Temperature scaling + 0–7 mapping, reliability curve, ECE | P5 | calibration report (dev-fitted) |
| P7 | Live app on Windows | P2,P6 | latency log, demo log vs reference |
| P8 [~] | Model B training + comparison table | P4 | A vs B table |
| P9 | Fine-tune on own recordings; Wiener A/B; error-analysis loop | P1,P5 | per-condition PER, Wiener decision |
| P10 | Optional: distillation B→A; lexicon beam search + LM | P8 | experiment report |

---

## 7. Proposed repo structure

```
phoneme-recognition/
  README.md  requirements.txt  requirements-colab.txt  .gitignore  pyproject.toml
  configs/        frontend.yaml  model_a.yaml  model_b.yaml  train.yaml  labels.json
  notebooks/      train_colab.ipynb
  src/phonrec/
    audio/        io.py  resample.py
    phonemize/    cmu.py  g2p.py  normalize_text.py
    frontend/     filters.py  noise.py  wiener.py  vad.py  mel.py  pcen.py  pitch.py  streaming.py
    data/         manifest.py  dataset.py  sampler.py  augment.py  librispeech.py
    models/       blocks.py  model_a.py  model_b.py
    training/     train.py  checkpoint.py
    decoding/     greedy.py  beam_lm.py (later)
    evaluation/   per.py  confusion.py  calibration.py  latency.py  error_analysis.py
    streaming/    live_app.py
  scripts/        record.py  build_manifest.py  make_splits.py  phonemize_dataset.py  extract_features.py  evaluate.py  run_live.py
  tests/          test_labels.py  test_phonemize.py  test_frontend.py  test_streaming_equivalence.py  test_models.py  test_ctc_decode.py
  data/           manifest.csv (paths only; audio ignored by git)
  checkpoints/    (git-ignored)
  reports/        model cards, benchmark + evaluation reports
```

---

## 8. Decisions log
- D1 (2026-11-01, accepted): Model A block widths 128->128, 128->128, 128->192, 192->192; convs bias-free + BatchNorm -> 134,569 params (+2.7% vs 131K). Model B -> 2,318,889 (+0.8% vs 2.3M).
- D2 (accepted): PER target to be proposed after the first LibriSpeech baseline, then frozen before own-data test.
- D3: Label ids blank=0, SIL=1, phonemes 2-40 (configs/labels.json, frozen).
- D4: SIL at utterance edges + explicit `<pause>` tokens in own transcripts; no SIL between ordinary words.
- D5: Normalisation = PCEN + global train-set mean/std (no per-utterance CMVN, not streamable).
- D6: Channels 0-39 PCEN, 40 = log2(F0/150 Hz) (0 if unvoiced), 41 = voicing (max normalised autocorrelation, past-only 40 ms).
- D7: Microphone for now = laptop built-in mic (spec says USB mic); Windows audio enhancements off. Recorded as `mic` in manifest.
- D8: HPF = 2nd-order Butterworth 80 Hz + DC blocker (pole 0.995); ~18 dB at 30 Hz.
- (pending) internal long-pause SIL rule for LibriSpeech (currently edges only).
- D11 (2026-11-10, assumption: spec does not define "confidence"): confidence of an emitted phoneme = its max
  posterior over its greedy CTC frame span; "correct" = aligned match vs reference (sub/ins = wrong), SIL excluded.
  Temperature T (log_softmax(lp/T); argmax and PER unchanged) is fit on DEV by minimising the binary NLL of
  (confidence, correct). Held-out check: fit on one half of dev speakers (crc32 of speaker id), ECE on the other.
- D12 (2026-11-10, assumption): 0-7 scale = floor(8p) clipped to 7, so level k claims P(correct) in [k/8,(k+1)/8);
  calibrated when each level's measured accuracy falls in its range (level table in calibration.json).

## 9. Progress log
- 2026-11-01: Spec analysed; plan created.
- 2026-11-01: Environment: Windows 11, Python 3.13.3, laptop mic, Colab via Google AI Plus (GPU tier to confirm), Drive 400 GB.
- 2026-11-01: P0 (labels, normaliser, lexicon, phonemizer), P2 (streaming front end), decoding + PER, audio I/O, recorder, Model A/B code written.
  Verified in Claude's Linux workspace (numpy/scipy, no torch): 41 tests pass; chunked == offline features (all chunk sizes, Wiener on/off); no future-sample dependence; pitch within 2% on 100-300 Hz harmonics; front end RTF ~0.011 single core.
  NOT yet verified: model tests (need torch), recorder on a real mic, CMUdict coverage of prompts. Repo copied to D:\Usman\phoneme-recognition.
- 2026-11-01: ON USER'S PC: 55/55 tests pass (torch CPU). Model A 134,569 params, Model B 2,318,889; measured receptive field 53 frames, 26 future frames.
  Recorder works: device "Microphone Array (AMD Audio Dev)" MME @44.1 kHz -> 16 kHz mono 16-bit OK. Test clip: noise floor ~-43 dBFS, speech ~-27..-35 dBFS (SNR ~10-15 dB, low).
  Colab: T4 GPU (and v5e-1 TPU, not used: CTC/variable shapes poor on XLA).
- 2026-11-01: P4 code written: sharded resumable feature store, offline audio augmentation (gain + coloured/real noise; aug1 variant),
  LibriSpeech manifest, build_features.py, dataset + bucket sampler + SpecAugment, train.py (AdamW, warmup+cosine, AMP, clip 5,
  CTC fp32, early stop on dev PER, atomic last/best checkpoints with full resume state), evaluate.py (PER S/D/I, per-phoneme,
  per-condition, confusion, latency, log-prob dump), configs/train_a.json + train_b.json, notebooks/train_colab.ipynb,
  scripts/smoke_test.py, make_code_zip.py. Store build + quick->full shard consistency tested here (45 pass).
  NOT yet run: training/eval code (needs torch) -> user runs smoke test on PC, then Colab.
- Decision D9: pretraining features built with Wiener ON; Wiener vs bypass A/B deferred to own-recording fine-tune stage (where noise exists).
- Decision D10: noise augmentation is applied to audio offline (cached 'aug1' copy) because features are cached; SpecAugment is on-the-fly.
- 2026-11-02: PC re-check: 59 tests pass; smoke_test.py PASSED (synthetic data, PER meaningless); front end RTF 0.0104/core (15 s clip).
  CLAUDE.md added. Notebook §2 hardened: download marked complete only after `gzip -t` (.done marker) with
  OpenSLR mirror fallback; extraction uses a marker so an interrupted tar is redone. phonrec_code.zip built (60 files, 69 KB).
  P4 started: zip upload + Colab sections 1-5 with QUICK=True. Note: QUICK = first 6000 train-clean-100 utterances
  sorted by id, so it covers only a subset of speakers (count to be read from the run). No Colab results yet.
- 2026-11-02: Colab run 1, §1 failed: zip not at My Drive/phonrec/phonrec_code.zip; the cell continued, and pytest then
  ran from /content and crawled all of Drive (4.5 min, unrelated tests). Fix: §1 stops with a clear error if the
  zip is missing (lists shallow matches elsewhere), asserts the unzip, and runs `pytest ... tests` only.
- 2026-11-10: FIRST REAL MODEL A RUN (Colab T4, QUICK=True). Read from checkpoints/best.pt (file dated 2026-11-04):
  134,569 params, 40/40 epochs, 4760 steps (119 steps/epoch at batch 48 -> ~5.7k train utts), Wiener on,
  train variants clean+aug1. Best dev-clean PER 0.4245 (greedy, SIL excluded) at epoch 39 = the LAST epoch
  (bad_epochs 0) -> not converged; more data/epochs should still help. history.csv / §1-3 logs (phonemize
  report, speaker count) not yet copied to the PC.
  Sanity check on own mic recording s000_spk01_0000 (laptop mic, 8.1 s, 26 ref phonemes): PER 0.679.
  ONE utterance only, not a metric; suggests a large mic/room domain gap (expected; fine-tune in P9).
  P4 exit evidence partly met (best ckpt yes; dev PER curve not yet on PC).
- 2026-11-10: Colab notebook outputs received (user's Downloads/train_colab.ipynb). Same QUICK features for A and B.
  Data: train-clean-100 28,539 utts / 251 spk; dev-clean 2703 / 40; test-clean 2620 / 40.
  Used: train 5678 of the first 6000, dev 2523 of 2703 (rest dropped: OOV, >25 s, or too short for CTC;
  the split per reason is in phonemize_report.json, not yet seen). All dev numbers below are on these 2523 utts.
  Model A (40 ep, 119 steps/ep, ~40 s/epoch on T4): dev PER 0.9715 (ep0) -> 0.6577 (ep3) -> 0.4934 (ep10)
  -> 0.4461 (ep19) -> 0.4245 (ep39, best). Train loss 1.635 vs dev loss 1.335 at ep39 (dev < train because
  train has SpecAugment + aug1 noise). Gains in the last 10 epochs ~0.004 -> close to converged for this data
  and LR schedule; more data is the next lever.
  P5 dev eval (§7, evaluate.py on best.pt): PER 0.4246, S 46,210 / D 25,311 / I 2,857 of 175,187 ref phonemes;
  RTF 0.0004 on T4 (GPU, not the PC). Top confusions: IH->AH 1652, AH->IH 1251, D->T 768, T->D 697, M->N 636,
  EH->AH 582, AE->AH 546, S->Z 517, Z->S 474 (vowel centralisation + voicing pairs). Deletions (25k) far exceed insertions (2.9k)
  -> model under-emits phonemes. logprobs.npz saved on Drive (reports/model_a_v1_dev/) for P6.
  P8 Model B (40 ep, 178 steps/ep at batch 32, ~64 s/epoch): best dev PER 0.2463 at ep39 (still improving slowly).
  A vs B on identical QUICK data: 0.4245 vs 0.2463 (from training-loop dev eval; §7 not yet run for B).
  Not run: §9 test (kept for the end). Warnings seen are harmless (float(loss) on a grad tensor; one
  "scheduler before optimizer" warning = first AMP step skipped by GradScaler).
- 2026-11-10: Drive files on PC: reports/model_a_v1_dev/{results.json, logprobs.npz (2523 utts, float16), history.csv},
  checkpoints/model_b_best.pt, reports/phonemize_report_{train,dev}.json.
  Phonemize: train 5678/6000 kept (322 OOV-dropped; 224 unique OOV), dev 2540/2703 (163 OOV-dropped; 110 unique);
  17 more dev utts dropped by length -> 2523. Extra (LibriSpeech) lexicon supplied only 735 dev / 2975 train word
  tokens and words like BOZZLE, HURSTWOOD are still OOV -> SUSPECT the Drive copy of librispeech-lexicon.txt is
  truncated (the notebook skips the download if any file exists). To check in Colab (wc -l, grep BOZZLE).
- 2026-11-10: P6 code: evaluation/calibration.py, per.align_ops/hyp_correct (align() refactored on top of it,
  same results), scripts/calibrate.py, tests/test_calibration.py (synthetic: recovers T0=2 within 1.6-2.5).
  66 tests pass. NOT yet run on real dev data: needs reports/model_a_v1_dev/per_utterance.jsonl (refs) from Drive.
- 2026-11-10: P6 RUN on Model A dev (reports/model_a_v1_calib/{calibration.json, reliability.png}).
  2523 dev utts, 152,723 emitted non-SIL phonemes, emitted-phoneme accuracy 0.6788.
  float16 dump: 23 utts differ from evaluate.py hyps by 23 of 152,733 tokens (top-2 ties after rounding); accepted (<0.1%).
  Held-out speaker half (fit 25 spk, held 15 spk): T fit = 0.995; ECE 0.0158 (T=1) -> 0.0150; NLL 0.5219 -> 0.5219.
  Final T (all dev) = 0.995 ~ 1: Model A is already well calibrated on LibriSpeech dev-clean (slightly
  UNDER-confident at mid levels: level 4 claims 0.50-0.625, mean conf 0.561, accuracy 0.588).
  Held-out level accuracy: L1 0.251 (n 529), L2 0.341, L3 0.459, L4 0.588, L5 0.705, L6 0.823, L7 0.935 (n 13,707);
  L0 empty. All inside their claimed ranges except L1 (0.251 vs <=0.25, at the edge).
  CAVEAT: valid for LibriSpeech-like audio only; own laptop-mic data needs its own dev-fitted T (P9).
  P6 exit evidence met for Model A. Model B not calibrated yet (needs §7 eval of model_b_v1 with --save-logprobs).
