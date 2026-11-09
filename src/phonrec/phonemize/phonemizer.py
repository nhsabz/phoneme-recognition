"""Transcript -> phoneme symbol sequence -> label ids.

Silence policy (spec: "silence is used at utterance edges and long pauses only"):
* SIL is added at the start and end of every utterance.
* A ``<pause>`` token in the transcript becomes SIL (for own recordings).
* Consecutive SILs are merged. No SIL between ordinary words.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..labels import LabelMap
from .lexicon import Lexicon
from .normalize_text import PAUSE_TOKEN, normalize_transcript


@dataclass
class PhonemizeResult:
    words: list[str]
    phones: list[str] | None          # None if any word was OOV
    oov_words: list[str] = field(default_factory=list)


class Phonemizer:
    def __init__(self, lexicon: Lexicon, label_map: LabelMap, edge_silence: bool = True):
        self.lex = lexicon
        self.lm = label_map
        self.edge_silence = edge_silence

    def phonemize(self, transcript: str) -> PhonemizeResult:
        words = normalize_transcript(transcript)
        sil = self.lm.silence
        phones: list[str] = [sil] if self.edge_silence else []
        oov: list[str] = []
        for w in words:
            if w == PAUSE_TOKEN:
                phones.append(sil)
                continue
            p = self.lex.lookup(w)
            if p is None:
                oov.append(w)
            else:
                phones.extend(p)
        if self.edge_silence:
            phones.append(sil)
        merged: list[str] = []
        for p in phones:
            if p == sil and merged and merged[-1] == sil:
                continue
            merged.append(p)
        if all(p == sil for p in merged):
            oov = oov or ["<EMPTY>"]
        return PhonemizeResult(words, None if oov else merged, oov)

    def encode(self, transcript: str) -> list[int] | None:
        r = self.phonemize(transcript)
        return None if r.phones is None else self.lm.encode(r.phones)
