"""Feature-store build + read (no PyTorch needed)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from phonrec.audio.io import write_wav
from phonrec.data.augment import NoiseBank, augment_audio, utt_rng
from phonrec.data.store import FeatureStore, ShardWriter, shard_done

ROOT = Path(__file__).resolve().parents[1]


def test_shard_roundtrip(tmp_path):
    w = ShardWriter(tmp_path / "train" / "clean", 0)
    a, b = np.random.rand(10, 42), np.random.rand(7, 42)
    w.add({"id": "x", "labels": [1, 5, 1]}, a)
    w.add({"id": "y", "labels": [1, 6, 1]}, b)
    w.close()
    assert shard_done(tmp_path / "train" / "clean", 0)
    st = FeatureStore(tmp_path, "train", ["clean"])
    assert st.ids == ["x", "y"] and st.n_frames("y") == 7
    np.testing.assert_allclose(st.get("y"), b, atol=1e-3)   # float16 storage
    assert st.labels("x") == [1, 5, 1]


def test_augment_is_deterministic_and_bounded():
    x = 0.3 * np.sin(np.arange(16000) / 5)
    bank = NoiseBank(None)
    y1, i1 = augment_audio(x, utt_rng("u1", "aug1"), bank)
    y2, i2 = augment_audio(x, utt_rng("u1", "aug1"), bank)
    y3, _ = augment_audio(x, utt_rng("u2", "aug1"), bank)
    np.testing.assert_array_equal(y1, y2)
    assert i1 == i2 and not np.allclose(y1, y3)
    assert np.abs(y1).max() <= 0.99 + 1e-6


def test_build_features_script(tmp_path, cmu_path):
    rows = []
    for i, text in enumerate(["the cat", "hello world", "the cat sat", "zyxqu cat"]):
        p = tmp_path / f"u{i}.wav"
        write_wav(p, 0.1 * np.random.default_rng(i).standard_normal(16000))
        rows.append({"recording_id": f"u{i}", "audio_path": str(p), "transcript": text, "split": "train",
                     "noise": "quiet"})
    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "feats"
    cmd = [sys.executable, str(ROOT / "scripts/build_features.py"), "--manifest", str(man), "--out", str(out),
           "--split", "train", "--cmudict", str(cmu_path), "--variants", "clean", "aug1",
           "--workers", "1", "--shard-size", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads((out / "train" / "phonemize_report.json").read_text())
    assert rep["kept"] == 3 and rep["oov_top"][0][0] == "ZYXQU"
    st = FeatureStore(out, "train", ["clean", "aug1"])
    assert st.ids == ["u0", "u1", "u2"]
    assert st.get("u0", "clean").shape == (98, 42)
    assert not np.allclose(st.get("u0", "clean"), st.get("u0", "aug1"))
    # resume: re-running must skip finished shards and not change anything
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0 and "shard" not in r2.stdout.split("[phonemize]")[1].split("done")[0]


def test_quick_then_full_build_is_consistent(tmp_path, cmu_path):
    rows = []
    for i, text in enumerate(["zyxqu", "the cat", "hello world", "the cat sat", "cat", "sat"]):
        p = tmp_path / f"u{i}.wav"
        write_wav(p, 0.1 * np.random.default_rng(i).standard_normal(8000))
        rows.append({"recording_id": f"u{i}", "audio_path": str(p), "transcript": text, "split": "train"})
    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "feats"
    base = [sys.executable, str(ROOT / "scripts/build_features.py"), "--manifest", str(man), "--out", str(out),
            "--split", "train", "--cmudict", str(cmu_path), "--workers", "1", "--shard-size", "2"]
    assert subprocess.run(base + ["--max-utts", "3"], capture_output=True).returncode == 0   # -> 2 shards
    assert FeatureStore(out, "train").ids == ["u1", "u2", "u3"]
    assert subprocess.run(base, capture_output=True).returncode == 0                         # full
    assert FeatureStore(out, "train").ids == ["u1", "u2", "u3", "u4", "u5"]                 # no gaps/dupes
