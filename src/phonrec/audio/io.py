"""WAV I/O and resampling shared by the recorder, dataset tools and live app.

Only the standard-library ``wave`` module is used for WAV files (16-bit PCM).
Resampling uses scipy's polyphase filter with a fixed design so that the
recorder and the live app produce identical 16 kHz audio from the same mic.
"""
from __future__ import annotations

import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 32767 / 32768) * 32768.0).round().astype(np.int16)


def write_wav(path: str | Path, x: np.ndarray, sr: int = TARGET_SR) -> None:
    pcm = x if x.dtype == np.int16 else float_to_int16(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns float32 mono in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM supported")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = data.reshape(-1, ch).astype(np.float32) / 32768.0
    return to_mono(x), sr


def check_wav_spec(path: str | Path) -> dict:
    with wave.open(str(path), "rb") as w:
        info = {"sr": w.getframerate(), "channels": w.getnchannels(),
                "bits": 8 * w.getsampwidth(), "seconds": w.getnframes() / w.getframerate()}
    info["ok"] = info["sr"] == TARGET_SR and info["channels"] == 1 and info["bits"] == 16
    return info


def level_report(x: np.ndarray) -> dict:
    """Peak / RMS in dBFS and simple warnings for the recorder."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    db = lambda v: 20 * np.log10(max(v, 1e-9))  # noqa: E731
    warn = []
    if peak >= 0.99:
        warn.append("CLIPPING: move back or lower input volume")
    if db(rms) < -45:
        warn.append("VERY QUIET: move closer or raise input volume")
    return {"peak_dbfs": db(peak), "rms_dbfs": db(rms), "warnings": warn}
