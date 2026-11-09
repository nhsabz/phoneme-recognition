"""Audio-level augmentation (applied BEFORE the front end, so the filters,
noise tracker and PCEN see realistic input): random gain and added noise.

Noise sources: coloured noise (white / pink / brown, slowly modulated) and, if
given, a directory of real noise WAVs (e.g. MUSAN noise, your own room noise).
Each utterance gets a deterministic RNG seeded from its id -> reproducible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def utt_rng(uid: str, salt: str = "") -> np.random.Generator:
    h = hashlib.sha256((uid + "|" + salt).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def coloured_noise(n: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    w = rng.standard_normal(n)
    if kind == "white":
        x = w
    else:
        f = np.fft.rfftfreq(n)
        f[0] = f[1] if n > 1 else 1.0
        shape = f ** (-0.5 if kind == "pink" else -1.0)
        x = np.fft.irfft(np.fft.rfft(w) * shape, n=n)
    # slow amplitude modulation (0.2-1 Hz) so the noise is not perfectly stationary
    t = np.arange(n) / 16000.0
    mod = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t + rng.uniform(0, 6.3))
    x = x * mod
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


class NoiseBank:
    def __init__(self, noise_dir: str | Path | None = None, max_files: int = 200):
        self.clips: list[np.ndarray] = []
        if noise_dir and Path(noise_dir).exists():
            from ..audio.io import read_wav, resample
            for p in sorted(Path(noise_dir).rglob("*.wav"))[:max_files]:
                x, sr = read_wav(p)
                x = resample(x, sr)
                if len(x) > 16000:
                    self.clips.append(x / (np.sqrt(np.mean(x ** 2)) + 1e-12))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.clips and rng.random() < 0.5:
            c = self.clips[rng.integers(len(self.clips))]
            reps = int(np.ceil(n / len(c))) + 1
            c = np.tile(c, reps)
            s = rng.integers(0, len(c) - n)
            return c[s: s + n]
        return coloured_noise(n, rng, ["white", "pink", "brown"][rng.integers(3)])


def augment_audio(x: np.ndarray, rng: np.random.Generator, bank: NoiseBank,
                  snr_db=(5.0, 25.0), gain_db=(-12.0, 6.0), p_noise: float = 0.8) -> tuple[np.ndarray, dict]:
    info = {}
    y = x.astype(np.float64)
    if rng.random() < p_noise:
        snr = rng.uniform(*snr_db)
        nz = bank.sample(len(y), rng)
        p_sig = np.mean(y ** 2) + 1e-12
        y = y + nz * np.sqrt(p_sig / 10 ** (snr / 10))
        info["snr_db"] = round(float(snr), 1)
    g = rng.uniform(*gain_db)
    y = y * 10 ** (g / 20)
    peak = np.abs(y).max()
    if peak > 0.99:
        y *= 0.99 / peak
    info["gain_db"] = round(float(g), 1)
    return y.astype(np.float32), info
