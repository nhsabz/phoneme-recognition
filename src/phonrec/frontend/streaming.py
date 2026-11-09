"""Streaming front end: audio chunks in -> 42-channel feature frames out.

The SAME class is used for training (whole utterance pushed at once) and for
live inference (small chunks). Every stage is causal and stateful, so the
output does not depend on how the audio is chunked (see
tests/test_streaming_equivalence.py) and never uses future samples.

Per frame t (frame covers samples [160 t, 160 t + 400)):
  1. DC blocker + 80 Hz Butterworth high-pass        (sample domain, IIR state)
  2. Hann window, 512-pt FFT -> power spectrum X (257 bins)
  3. minimum-statistics noise floor N per bin        (recursive state)
  4. Wiener gain G (decision-directed SNR, floor) or bypass (G = 1)
  5. 40 triangular mel bands 80-7600 Hz on G^2 X
  6. PCEN (or log-mel)                               (recursive smoother state)
  7. pitch on the past 40 ms ending at the frame end: log2(F0/150) and voicing
  8. soft VAD probability (side output, not a feature)
Channel layout: 0-39 PCEN/log-mel, 40 = log-F0 (0 if unvoiced), 41 = voicing.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .config import FrontendConfig
from .mel import mel_filterbank
from .pitch import pitch_features


class StreamingFrontend:
    def __init__(self, cfg: FrontendConfig | None = None):
        self.cfg = cfg = cfg or FrontendConfig()
        sr = cfg.sample_rate
        hp = butter(cfg.hpf_order, cfg.hpf_hz, btype="highpass", fs=sr, output="sos")
        dc = np.array([[1.0, -1.0, 0.0, 1.0, -cfg.dc_pole, 0.0]])
        self.sos = np.vstack([dc, hp])
        self.window = np.hanning(cfg.win_length + 1)[:-1].astype(np.float64)  # periodic Hann
        self.mel = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
        self.vad_band = (freqs >= cfg.vad_lo_hz) & (freqs <= cfg.vad_hi_hz)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        c = self.cfg
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.lead = c.pitch_win - c.win_length          # history needed before a frame
        self.buf = np.zeros(self.lead)                  # zeros before the first sample
        self.buf_start = -self.lead                     # global index of buf[0]
        self.next_frame = 0
        # min statistics
        self.ms_P = None
        self.ms_cur = None
        self.ms_count = 0
        self.ms_subs: deque = deque(maxlen=c.ms_num_sub)
        # wiener
        self.prev_G = None
        self.prev_X = None
        # pcen smoother
        self.pcen_M = None
        # vad
        self.vad_p = 0.0

    # ------------------------------------------------------------------- API
    def push(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """audio: float in [-1, 1] (or int16). Returns features (n, 42) and vad (n,)."""
        c = self.cfg
        x = np.asarray(audio)
        if x.dtype == np.int16:
            x = x.astype(np.float64) / 32768.0
        x = x.astype(np.float64).reshape(-1) * c.input_scale
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        self.buf = np.concatenate([self.buf, y])

        # complete frames available: frame t needs global samples up to 160 t + 400
        buf_end = self.buf_start + len(self.buf)
        n = 0
        if buf_end >= c.win_length:
            n = max(0, (buf_end - c.win_length) // c.hop_length + 1 - self.next_frame)
        if n == 0:
            return {"features": np.zeros((0, c.num_channels), np.float32), "vad": np.zeros(0, np.float32)}

        t0 = self.next_frame
        starts = c.hop_length * np.arange(t0, t0 + n)            # global start of each frame
        ends = starts + c.win_length
        rel_p = (ends - c.pitch_win) - self.buf_start           # pitch window start in buf
        idx = rel_p[:, None] + np.arange(c.pitch_win)[None, :]
        pwin = self.buf[idx]                                    # (n, 640)
        frames = pwin[:, self.lead:]                            # (n, 400) analysis frames

        X = np.abs(np.fft.rfft(frames * self.window, n=c.n_fft, axis=1)) ** 2  # (n, 257)
        G = np.empty_like(X)
        vad = np.empty(n)
        for i in range(n):
            N = self._noise_update(X[i])
            G[i] = self._wiener(X[i], N) if c.wiener else 1.0
            vad[i] = self._vad(X[i], N)
        E = (G * G * X) @ self.mel.T                            # (n, 40)
        spec = self._pcen(E) if c.feature == "pcen" else np.log(E + 1.0)
        f0f, voi = pitch_features(pwin, c)

        self.next_frame += n
        keep_from = c.hop_length * self.next_frame - self.lead  # earliest sample still needed
        drop = keep_from - self.buf_start
        if drop > 0:
            self.buf = self.buf[drop:]
            self.buf_start = keep_from
        feats = np.concatenate([spec, f0f[:, None], voi[:, None]], axis=1)
        return {"features": feats.astype(np.float32), "vad": vad.astype(np.float32)}

    # -------------------------------------------------------------- stages
    def _noise_update(self, X: np.ndarray) -> np.ndarray:
        c = self.cfg
        if self.ms_P is None:
            self.ms_P = X.copy()
            self.ms_cur = X.copy()
        else:
            self.ms_P = c.ms_alpha * self.ms_P + (1 - c.ms_alpha) * X
            np.minimum(self.ms_cur, self.ms_P, out=self.ms_cur)
        m = self.ms_cur
        for s in self.ms_subs:
            m = np.minimum(m, s)
        self.ms_count += 1
        if self.ms_count == c.ms_sub_frames:
            self.ms_subs.append(self.ms_cur.copy())
            self.ms_cur = self.ms_P.copy()
            self.ms_count = 0
        return np.maximum(c.ms_bias * m, 1e-10)

    def _wiener(self, X: np.ndarray, N: np.ndarray) -> np.ndarray:
        c = self.cfg
        gamma = X / N                                            # a-posteriori SNR
        ml = np.maximum(gamma - 1.0, 0.0)
        if self.prev_G is None:
            xi = ml
        else:
            xi = c.dd_alpha * (self.prev_G ** 2) * self.prev_X / N + (1 - c.dd_alpha) * ml
        G = np.maximum(xi / (1.0 + xi), c.gain_floor)
        self.prev_G, self.prev_X = G, X
        return G

    def _vad(self, X: np.ndarray, N: np.ndarray) -> float:
        c = self.cfg
        snr_db = 10 * np.log10((X[self.vad_band].sum() + 1e-10) / (N[self.vad_band].sum() + 1e-10))
        p_raw = 1.0 / (1.0 + np.exp(-(snr_db - c.vad_snr_db) / c.vad_slope_db))
        a = c.vad_attack if p_raw > self.vad_p else c.vad_release
        self.vad_p += a * (p_raw - self.vad_p)
        return self.vad_p

    def _pcen(self, E: np.ndarray) -> np.ndarray:
        c = self.cfg
        if self.pcen_M is None:
            self.pcen_M = E[0].copy()
        s = c.pcen_s
        M, zf = lfilter([s], [1.0, -(1 - s)], E, axis=0, zi=((1 - s) * self.pcen_M)[None, :])
        self.pcen_M = M[-1].copy()
        return (E / (c.pcen_eps + M) ** c.pcen_alpha + c.pcen_delta) ** c.pcen_r - c.pcen_delta ** c.pcen_r


def num_frames(n_samples: int, cfg: FrontendConfig | None = None) -> int:
    cfg = cfg or FrontendConfig()
    return 0 if n_samples < cfg.win_length else 1 + (n_samples - cfg.win_length) // cfg.hop_length


def extract_features(audio: np.ndarray, cfg: FrontendConfig | None = None) -> dict[str, np.ndarray]:
    """Offline extraction for one utterance (fresh state, identical to live)."""
    return StreamingFrontend(cfg).push(audio)
