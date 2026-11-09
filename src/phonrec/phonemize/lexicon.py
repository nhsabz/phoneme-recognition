"""Pronunciation lexicon (CMUdict format) with layered lookup and OOV reporting.

Lookup order for each word:
  1. manual overrides  (lexicon/overrides.txt; always wins)
  2. CMUdict           (first pronunciation only, deterministic)
  3. extra lexicons    (e.g. LibriSpeech's librispeech-lexicon.txt)
  4. optional G2P      (g2p_en, only if installed and enabled)
  5. otherwise OOV     (word reported; utterance can be dropped)

Accepted file formats (one entry per line):
  ABANDON  AH0 B AE1 N D AH0 N       (cmudict-0.7b, ';;;' comments)
  abandon AH0 B AE1 N D AH0 N        (cmusphinx cmudict.dict, '#' comments)
  ABANDON(2)  ...                    (variants are ignored; first entry kept)
  ABANDON\tAH0 B AE1 N D AH0 N       (LibriSpeech lexicon, tab separated)
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..labels import PHONEMES_39

_VALID = set(PHONEMES_39)
_VARIANT = re.compile(r"\(\d+\)$")


def strip_stress(phones: list[str]) -> list[str]:
    out = []
    for p in phones:
        p = re.sub(r"\d", "", p.upper())
        if p not in _VALID:
            raise ValueError(f"invalid phoneme {p!r}")
        out.append(p)
    return out


def load_lexicon_file(path: str | Path) -> dict[str, list[str]]:
    lex: dict[str, list[str]] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].upper()
            if _VARIANT.search(word):
                continue  # keep first pronunciation only
            if word in lex:
                continue
            try:
                lex[word] = strip_stress(parts[1:])
            except ValueError:
                continue  # malformed entry: skip, never crash
    return lex


def load_cmudict_package() -> dict[str, list[str]]:
    """CMUdict from the ``cmudict`` pip package (first pronunciation, stress removed)."""
    import cmudict  # pip install cmudict

    lex: dict[str, list[str]] = {}
    for word, prons in cmudict.dict().items():
        if not prons:
            continue
        try:
            lex[word.upper()] = strip_stress(prons[0])
        except ValueError:
            continue
    return lex


class Lexicon:
    def __init__(self, cmudict: dict[str, list[str]],
                 overrides: dict[str, list[str]] | None = None,
                 extra: list[dict[str, list[str]]] | None = None,
                 use_g2p: bool = False):
        self.overrides = overrides or {}
        self.cmudict = cmudict
        self.extra = extra or []
        self.oov = Counter()
        self.source = Counter()
        self._g2p = None
        if use_g2p:
            try:
                from g2p_en import G2p  # optional dependency
                self._g2p = G2p()
            except Exception as e:  # noqa: BLE001
                print(f"[lexicon] g2p_en unavailable ({e}); OOV words will be reported only")

    @classmethod
    def from_files(cls, cmudict_path, overrides_path=None, extra_paths=(), use_g2p=False):
        over = load_lexicon_file(overrides_path) if overrides_path and Path(overrides_path).exists() else {}
        cmu = load_cmudict_package() if cmudict_path in (None, "package") else load_lexicon_file(cmudict_path)
        return cls(cmu, over,
                   [load_lexicon_file(p) for p in extra_paths], use_g2p)

    def lookup(self, word: str) -> list[str] | None:
        w = word.upper()
        for name, table in (("override", self.overrides), ("cmudict", self.cmudict)):
            if w in table:
                self.source[name] += 1
                return table[w]
        for table in self.extra:
            if w in table:
                self.source["extra"] += 1
                return table[w]
        if self._g2p is not None and re.fullmatch(r"[A-Z']+", w):
            phones = [p for p in self._g2p(w.lower()) if re.sub(r"\d", "", p) in _VALID]
            if phones:
                self.source["g2p"] += 1
                return strip_stress(phones)
        self.oov[w] += 1
        return None
