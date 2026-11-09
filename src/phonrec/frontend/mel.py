"""HTK-style triangular mel filterbank (no librosa dependency)."""
from __future__ import annotations

import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Returns (n_mels, n_fft//2+1) weights; band edges equally spaced on the mel scale."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, ce, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb
