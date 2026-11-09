"""Phoneme error rate with substitution / deletion / insertion breakdown and alignment."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AlignStats:
    n_ref: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0
    # (ref, hyp) pairs; '' marks a gap. Correct matches are included.
    pairs: Counter = field(default_factory=Counter)

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def per(self) -> float:
        return self.errors / max(self.n_ref, 1)

    def __iadd__(self, o: "AlignStats"):
        self.n_ref += o.n_ref
        self.sub += o.sub
        self.dele += o.dele
        self.ins += o.ins
        self.pairs.update(o.pairs)
        return self

    def as_dict(self) -> dict:
        return {"n_ref": self.n_ref, "sub": self.sub, "del": self.dele, "ins": self.ins,
                "errors": self.errors, "per": self.per}


def align(ref: list, hyp: list) -> AlignStats:
    """Levenshtein alignment (unit costs) with backtrace."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
    st = AlignStats(n_ref=n)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                st.sub += 1
            st.pairs[(ref[i - 1], hyp[j - 1])] += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            st.dele += 1
            st.pairs[(ref[i - 1], "")] += 1
            i -= 1
        else:
            st.ins += 1
            st.pairs[("", hyp[j - 1])] += 1
            j -= 1
    return st


def corpus_per(refs: list[list], hyps: list[list]) -> AlignStats:
    total = AlignStats()
    for r, h in zip(refs, hyps, strict=True):
        total += align(r, h)
    return total
