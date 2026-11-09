"""Global per-channel mean/variance normalisation.

Per-utterance CMVN would need the whole utterance (future context) and so
cannot be used live. Instead mean/std are computed once on the TRAINING split
and stored next to the model; training, evaluation and live inference apply
the same fixed constants.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)
        self._n = 0
        self._s = None
        self._ss = None

    def accumulate(self, feats: np.ndarray):
        f = feats.astype(np.float64)
        if self._s is None:
            self._s = np.zeros(f.shape[1])
            self._ss = np.zeros(f.shape[1])
        self._n += len(f)
        self._s += f.sum(0)
        self._ss += (f * f).sum(0)

    def finalize(self):
        m = self._s / self._n
        v = np.maximum(self._ss / self._n - m * m, 1e-8)
        self.mean, self.std = m.astype(np.float32), np.sqrt(v).astype(np.float32)
        return self

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["std"])
