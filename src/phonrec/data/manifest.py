"""Unified utterance manifest (CSV). Same columns for own recordings and LibriSpeech.

recording_id, audio_path, transcript, speaker_id, session_id, noise, distance, mic,
device, native_sr, duration_s, split, created
audio_path may be relative to the manifest's base directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ["recording_id", "audio_path", "transcript", "speaker_id", "session_id", "noise",
          "distance", "mic", "device", "native_sr", "duration_s", "split", "created"]


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
