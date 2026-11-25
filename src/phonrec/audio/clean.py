"""Clean own recordings before feature extraction (data cleaning, not part of the front end).

Measured on session s001 (2026-11-25, laptop mic, scripts/record.py):
* every take starts with 240-260 ms of digital silence (~-101 dBFS) while the mic stream opens;
  the front end's minimum-statistics noise tracker would then believe the room is silent for ~1.5 s
  (its search window), so the Wiener filter barely acts on a third of a typical take;
* 21 of 40 takes end with a 10-110 ms unvoiced burst after a quiet gap: the Enter key that stops
  the recording (two more takes end loud for >= 250 ms, i.e. speech, and are kept).
Neither exists in LibriSpeech or in a running live stream, so both are removed:
* leading 10 ms frames below ``silence_dbfs`` (digital silence, far below any room noise);
* a burst in the last ``max_click_ms`` (peak >= ``click_db`` above the room floor) that rises
  straight after >= ``quiet_ms`` within 10 dB of the floor; it and anything after it are cut.
  A loud stretch longer than ``max_click_ms`` is treated as speech and kept.
Decision D14 in phoneme-pipeline-plan.md.
"""
from __future__ import annotations

import numpy as np

FRAME = 160   # 10 ms at 16 kHz


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0)
    fr = np.asarray(x[: n * FRAME], np.float64).reshape(n, FRAME)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


def trim_recording(x: np.ndarray, silence_dbfs: float = -90.0, click_db: float = 15.0,
                   max_click_ms: int = 150, quiet_ms: int = 50) -> tuple[np.ndarray, dict]:
    """x: 16 kHz float audio. Returns (trimmed audio, {"lead_ms", "tail_ms"})."""
    db = frame_db(x)
    if len(db) == 0:
        return x, {"lead_ms": 0, "tail_ms": 0}
    loud = np.flatnonzero(db > silence_dbfs)
    lead = int(loud[0]) if len(loud) else 0
    db = db[lead:]
    floor = np.percentile(db, 10) if len(db) else 0.0
    win, q, tail = max_click_ms // 10, quiet_ms // 10, 0
    peaks = np.flatnonzero(db[-win:] > floor + click_db)    # loud frames in the last max_click_ms
    if len(peaks):
        start = len(db) - min(win, len(db)) + int(peaks[0])
        while start > 0 and db[start - 1] > floor + 10:     # back to where the burst rises
            start -= 1
        before = db[start - q: start] if start >= q else db[:0]
        if len(db) - start <= win and len(before) == q and (before < floor + 10).all():
            tail = len(db) - start                          # cut the click and everything after it
    end = len(x) - tail * FRAME if tail else len(x)
    return x[lead * FRAME: end], {"lead_ms": lead * 10, "tail_ms": tail * 10}
