"""Pitch (F0) and voicing strength from a past-only window.

For each frame we take the 40 ms of high-passed audio that ENDS at the frame's
last sample (no future samples) and compute the normalised autocorrelation
    nacf(l) = r(l) / r(0) * N / (N - l)
over lags for 60-400 Hz. Voicing = max nacf in [0, 1]. To avoid sub-harmonic
(octave-down) errors the SHORTEST lag reaching 90 % of the maximum is taken,
refined by parabolic interpolation.
Channels:  log2(F0 / 150 Hz) when voicing >= threshold, else 0;  voicing.
"""
from __future__ import annotations

import numpy as np


def pitch_features(win: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    n, N = win.shape
    sr = cfg.sample_rate
    lag_min = int(np.floor(sr / cfg.f0_max))
    lag_max = int(np.ceil(sr / cfg.f0_min))
    nfft = 1 << int(np.ceil(np.log2(N + lag_max + 1)))
    x = win - win.mean(axis=1, keepdims=True)
    F = np.fft.rfft(x, n=nfft, axis=1)
    r = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, : lag_max + 2]
    r0 = r[:, :1]
    lags = np.arange(r.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        nacf = r / np.maximum(r0, 1e-12) * (N / np.maximum(N - lags, 1))[None, :]
    nacf = np.clip(np.nan_to_num(nacf), -1.0, 1.0)

    seg = nacf[:, lag_min: lag_max + 1]
    vmax = seg.max(axis=1)
    # shortest lag reaching 90 % of the peak
    ok = seg >= 0.9 * vmax[:, None]
    k = ok.argmax(axis=1)
    # walk to the local peak at/after k
    for _ in range(4):
        nxt = np.minimum(k + 1, seg.shape[1] - 1)
        better = seg[np.arange(n), nxt] > seg[np.arange(n), k]
        k = np.where(better, nxt, k)
    lag = (k + lag_min).astype(np.float64)
    li = np.clip(k + lag_min, 1, nacf.shape[1] - 2)
    a, b, c = nacf[np.arange(n), li - 1], nacf[np.arange(n), li], nacf[np.arange(n), li + 1]
    den = a - 2 * b + c
    safe = np.abs(den) > 1e-9
    delta = np.zeros(n)
    delta[safe] = 0.5 * (a[safe] - c[safe]) / den[safe]
    lag = lag + np.clip(delta, -0.5, 0.5)

    energy = r0[:, 0] / N
    voicing = np.where(energy > 1e-3, np.clip(vmax, 0.0, 1.0), 0.0)
    f0 = sr / np.maximum(lag, 1.0)
    f0_feat = np.where(voicing >= cfg.voicing_threshold, np.log2(f0 / cfg.f0_ref_hz), 0.0)
    return f0_feat, voicing


def estimate_f0(win: np.ndarray, cfg) -> np.ndarray:
    """Helper for tests: F0 in Hz (0 where unvoiced)."""
    f0f, v = pitch_features(win, cfg)
    return np.where(v >= cfg.voicing_threshold, cfg.f0_ref_hz * 2.0 ** f0f, 0.0)
