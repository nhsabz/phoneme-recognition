"""Temperature scaling and the 0-7 phoneme confidence scale (fit on dev only, never on test).

Confidence of an emitted phoneme = its highest posterior over the frames that emitted it
(greedy CTC spans, see decoding.greedy). Correct = aligned as a match to the reference;
substitutions and insertions are wrong. SIL is stripped from both sides, as in PER.

Temperature: lp_T = log_softmax(lp / T). Dividing log-softmax outputs by T equals dividing the
logits by T (the per-frame constant cancels), and the argmax does not move, so greedy hypotheses,
and therefore PER, are unchanged; only confidences change. T is chosen to minimise the binary NLL
of (confidence, correct) on dev, i.e. it calibrates exactly the number the live app shows.

0-7 scale: level k = floor(8 p), clipped to 7. Level k claims P(correct) in [k/8, (k+1)/8).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from ..decoding.greedy import greedy_decode
from .per import hyp_correct

N_LEVELS = 8


def apply_temperature(lp: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(lp, dtype=np.float32) / T
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def token_confidences(lp: np.ndarray, T: float = 1.0, blank_id: int = 0):
    """Greedy tokens of one utterance with their confidence. Returns (ids, confs, spans)."""
    ids, spans = greedy_decode(lp, blank_id, return_frames=True)
    p = np.exp(apply_temperature(lp, T))
    return ids, [float(p[a:b, k].max()) for k, (a, b) in zip(ids, spans)], spans


@dataclass
class TokenSet:
    """All emitted (non-SIL) dev tokens: the frame rows that emitted each one, its class,
    whether it was correct, and which utterance it came from."""
    rows: np.ndarray      # (n_rows, C) float32 log-probs at T = 1
    starts: np.ndarray    # (n_tok,) first row of each token in `rows`
    cls: np.ndarray       # (n_tok,) label id
    correct: np.ndarray   # (n_tok,) bool
    uid: np.ndarray       # (n_tok,) utterance id

    def subset(self, mask: np.ndarray) -> "TokenSet":
        ends = np.append(self.starts[1:], len(self.rows))
        idx = np.flatnonzero(mask)
        pieces = [self.rows[self.starts[i]: ends[i]] for i in idx]
        lens = np.array([len(x) for x in pieces], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(idx) else np.zeros(0, np.int64)
        rows = np.concatenate(pieces) if pieces else self.rows[:0]
        return TokenSet(rows, starts, self.cls[idx], self.correct[idx], self.uid[idx])

    def confidences(self, T: float) -> np.ndarray:
        if len(self.cls) == 0:
            return np.zeros(0)
        lens = np.diff(np.append(self.starts, len(self.rows)))
        p = np.exp(apply_temperature(self.rows, T))[np.arange(len(self.rows)), np.repeat(self.cls, lens)]
        return np.maximum.reduceat(p, self.starts)


def collect_tokens(logprobs: dict, refs: dict, blank_id: int = 0, silence_id: int = 1) -> TokenSet:
    """logprobs: uid -> (T, C) log-probs; refs: uid -> reference label ids (SIL already stripped)."""
    rows, starts, cls, ok, uids = [], [], [], [], []
    n = 0
    for uid in sorted(refs):
        lp = np.asarray(logprobs[uid], dtype=np.float32)
        ids, spans = greedy_decode(lp, blank_id, return_frames=True)
        keep = [(k, s) for k, s in zip(ids, spans) if k != silence_id]
        hyp = [k for k, _ in keep]
        for (k, (a, b)), good in zip(keep, hyp_correct(list(refs[uid]), hyp)):
            starts.append(n)
            rows.append(lp[a:b])
            n += b - a
            cls.append(k)
            ok.append(good)
            uids.append(uid)
    C = next(iter(logprobs.values())).shape[-1] if logprobs else 41
    return TokenSet(np.concatenate(rows) if rows else np.zeros((0, C), np.float32),
                    np.asarray(starts, np.int64), np.asarray(cls, np.int64),
                    np.asarray(ok, bool), np.asarray(uids))


def binary_nll(p: np.ndarray, correct: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(np.where(correct, np.log(p), np.log1p(-p))))


def fit_temperature(ts: TokenSet, lo: float = 0.25, hi: float = 8.0) -> float:
    """Coarse log-spaced grid, then a finer grid around the best point."""
    def loss(T):
        return binary_nll(ts.confidences(T), ts.correct)
    grid = np.geomspace(lo, hi, 41)
    best = grid[int(np.argmin([loss(T) for T in grid]))]
    fine = np.geomspace(best / 1.1, best * 1.1, 41)
    return float(fine[int(np.argmin([loss(T) for T in fine]))])


def to_level(p) -> np.ndarray:
    return np.minimum(N_LEVELS - 1, np.floor(np.asarray(p) * N_LEVELS)).astype(np.int64)


def level_table(p: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Per 0-7 level: how many tokens, mean confidence, measured accuracy (the reliability curve)."""
    lv = to_level(p)
    out = []
    for k in range(N_LEVELS):
        m = lv == k
        out.append({"level": k, "claims": [k / N_LEVELS, (k + 1) / N_LEVELS], "n": int(m.sum()),
                    "mean_conf": float(p[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def ece(p: np.ndarray, correct: np.ndarray) -> float:
    """Expected calibration error over the 8 equal-width bins of the 0-7 scale."""
    lv = to_level(p)
    tot = 0.0
    for k in range(N_LEVELS):
        m = lv == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - correct[m].mean())
    return float(tot / max(len(p), 1))


def speaker_half(uid: str) -> int:
    """Deterministic speaker-level split of LibriSpeech ids ('spk-chapter-utt') into halves 0 / 1."""
    return zlib.crc32(uid.split("-")[0].encode()) % 2
