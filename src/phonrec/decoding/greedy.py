"""Greedy (best-path) CTC decoding.

1. argmax class per output frame
2. merge consecutive identical labels
3. remove blanks
A phoneme repeated with a blank in between (e.g. ``T _ T``) is kept twice;
a run without a blank (``T T``) collapses to one -- standard CTC semantics.
SIL is an ordinary class here; scoring decides whether to keep it.
"""
from __future__ import annotations

import numpy as np


def collapse(path, blank_id: int = 0) -> list[int]:
    out: list[int] = []
    prev = None
    for k in path:
        k = int(k)
        if k != prev and k != blank_id:
            out.append(k)
        prev = k
    return out


def greedy_decode(log_probs: np.ndarray, blank_id: int = 0, return_frames: bool = False):
    """log_probs: (T, C) for one utterance. Returns ids (and per-token frame spans).

    Frame spans let callers compute phoneme-level confidence from the frames
    that emitted each token.
    """
    lp = np.asarray(log_probs)
    path = lp.argmax(axis=-1)
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    prev = None
    for t, k in enumerate(path):
        k = int(k)
        if k != blank_id:
            if k != prev:
                ids.append(k)
                spans.append((t, t + 1))
            else:
                spans[-1] = (spans[-1][0], t + 1)
        prev = k
    return (ids, spans) if return_frames else ids


def strip_silence(ids, silence_id: int = 1) -> list[int]:
    return [i for i in ids if i != silence_id]
