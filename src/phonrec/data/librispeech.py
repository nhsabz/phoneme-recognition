"""LibriSpeech (OpenSLR 12, CC BY 4.0) -> unified manifest rows.

Expects the extracted layout  <root>/LibriSpeech/<subset>/<speaker>/<chapter>/<spk>-<chap>.trans.txt
Official subsets are speaker-disjoint, so train/dev/test splits do not leak speakers.
"""
from __future__ import annotations

from pathlib import Path

SUBSET_SPLIT = {"train-clean-100": "train", "train-clean-360": "train", "train-other-500": "train",
                "dev-clean": "dev", "dev-other": "dev", "test-clean": "test", "test-other": "test"}


def librispeech_rows(root: str | Path, subset: str) -> list[dict]:
    base = Path(root) / "LibriSpeech" / subset
    if not base.exists():
        raise FileNotFoundError(base)
    rows = []
    for trans in sorted(base.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            spk, chap, _ = uid.split("-")
            rows.append({
                "recording_id": uid, "audio_path": str(trans.parent / f"{uid}.flac"), "transcript": text,
                "speaker_id": f"ls{spk}", "session_id": f"ls{spk}-{chap}", "noise": "clean",
                "distance": "", "mic": "librispeech", "split": SUBSET_SPLIT.get(subset, subset)})
    return rows
