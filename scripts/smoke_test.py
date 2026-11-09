"""End-to-end smoke test on tiny synthetic data (CPU, ~1-3 minutes).

Checks that the whole chain runs: manifest -> phonemize -> features (clean + aug1)
-> train Model A for 2 epochs -> resume to 3 epochs -> evaluate -> train Model B 1 epoch.
The audio is synthetic (tone sequences), so the PER numbers are meaningless;
only "does it run, does the loss go down, does resume work" is tested.

  python scripts/smoke_test.py            (keeps output in ./smoke_out)
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.audio.io import write_wav  # noqa: E402

LEX = """HELLO  HH AH0 L OW1
WORLD  W ER1 L D
THE  DH AH0
CAT  K AE1 T
SAT  S AE1 T
TOP  T AA1 P
PAT  P AE1 T
DOG  D AO1 G
"""
WORDS = ["HELLO", "WORLD", "THE", "CAT", "SAT", "TOP", "PAT", "DOG"]


def synth(words, rng):
    """Each word -> a distinct tone pattern, so the task is learnable."""
    sr, out = 16000, [np.zeros(3200)]
    for w in words:
        k = WORDS.index(w)
        t = np.arange(int(0.35 * sr)) / sr
        out.append(0.2 * np.sin(2 * np.pi * (200 + 120 * k) * t) * np.hanning(len(t)))
        out.append(0.002 * rng.standard_normal(800))
    out.append(np.zeros(3200))
    return np.concatenate(out) + 0.001 * rng.standard_normal(sum(len(o) for o in out))


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-5000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def main(out_dir: str = "smoke_out"):
    out = ROOT / out_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "wav").mkdir(parents=True)
    (out / "lex.txt").write_text(LEX)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(48):
        words = list(rng.choice(WORDS, size=rng.integers(2, 5)))
        p = out / "wav" / f"u{i:03d}.wav"
        write_wav(p, synth(words, rng))
        rows.append({"recording_id": f"u{i:03d}", "audio_path": str(p), "transcript": " ".join(words).lower(),
                     "speaker_id": "synth", "session_id": f"s{i % 4}", "noise": "quiet",
                     "distance": "near", "split": "dev" if i >= 40 else "train"})
    man = out / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    py = sys.executable
    feats = out / "features"
    common = ["--manifest", str(man), "--out", str(feats), "--cmudict", str(out / "lex.txt"),
              "--filter-split", "--workers", "1", "--shard-size", "16"]
    run([py, "scripts/build_features.py", *common, "--split", "train", "--variants", "clean", "aug1"])
    run([py, "scripts/build_features.py", *common, "--split", "dev", "--variants", "clean"])

    cfg = json.loads((ROOT / "configs/train_a.json").read_text())
    cfg.update({"epochs": 2, "batch_size": 8, "num_workers": 0, "lr": 3e-3, "log_every": 5, "warmup_epochs": 0.2})
    (out / "cfg_a.json").write_text(json.dumps(cfg))
    run_a = out / "run_a"
    run([py, "-m", "phonrec.training.train", "--config", str(out / "cfg_a.json"), "--features", str(feats),
         "--out", str(run_a), "--device", "cpu"])
    cfg["epochs"] = 3
    (out / "cfg_a.json").write_text(json.dumps(cfg))
    log = run([py, "-m", "phonrec.training.train", "--config", str(out / "cfg_a.json"), "--features", str(feats),
               "--out", str(run_a), "--device", "cpu"])
    assert "resumed from" in log and "at epoch 2" in log, "resume did not work"
    hist = list(csv.DictReader(open(run_a / "history.csv")))
    assert [int(h["epoch"]) for h in hist] == [0, 1, 2], hist
    assert float(hist[-1]["train_loss"]) < float(hist[0]["train_loss"]), "train loss did not decrease"
    run([py, "-m", "phonrec.evaluation.evaluate", "--ckpt", str(run_a / "best.pt"), "--features", str(feats),
         "--split", "dev", "--out", str(out / "eval_a"), "--device", "cpu"])
    res = json.loads((out / "eval_a" / "results.json").read_text())
    assert res["n_utts"] == 8 and res["n_params"] == 134_569

    cfg_b = json.loads((ROOT / "configs/train_b.json").read_text())
    cfg_b.update({"epochs": 1, "batch_size": 8, "num_workers": 0, "log_every": 5})
    (out / "cfg_b.json").write_text(json.dumps(cfg_b))
    run([py, "-m", "phonrec.training.train", "--config", str(out / "cfg_b.json"), "--features", str(feats),
         "--out", str(out / "run_b"), "--device", "cpu"])
    print("\nSMOKE TEST PASSED: train loss", hist[0]["train_loss"], "->", hist[-1]["train_loss"])


if __name__ == "__main__":
    main()
