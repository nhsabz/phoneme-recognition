"""Frozen label map: CTC blank, silence and the 39 CMUdict phonemes (41 classes)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[2] / "configs" / "labels.json"

PHONEMES_39 = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()


@dataclass(frozen=True)
class LabelMap:
    labels: tuple[str, ...]
    blank: str = "<blank>"
    silence: str = "SIL"

    def __post_init__(self):
        if len(self.labels) != 41:
            raise ValueError(f"expected 41 labels, got {len(self.labels)}")
        if len(set(self.labels)) != 41:
            raise ValueError("duplicate labels")
        if self.labels[0] != self.blank:
            raise ValueError("CTC blank must be id 0")
        if self.labels[1] != self.silence:
            raise ValueError("silence must be id 1")
        if sorted(self.labels[2:]) != sorted(PHONEMES_39):
            raise ValueError("ids 2-40 must be exactly the 39 CMUdict phonemes")

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def silence_id(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def to_id(self, sym: str) -> int:
        try:
            return self.labels.index(sym)
        except ValueError as e:
            raise KeyError(f"unknown label {sym!r}") from e

    def encode(self, syms: list[str]) -> list[int]:
        ids = [self.to_id(s) for s in syms]
        if self.blank_id in ids:
            raise ValueError("blank must never appear in a target sequence")
        return ids

    def decode(self, ids) -> list[str]:
        return [self.labels[int(i)] for i in ids]


def load_label_map(path: str | Path = DEFAULT_LABELS_PATH) -> LabelMap:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return LabelMap(tuple(d["labels"]), d.get("blank", "<blank>"), d.get("silence", "SIL"))
