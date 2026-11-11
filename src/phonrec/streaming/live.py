"""Live (streaming) inference: mic chunks -> phonemes with a 0-7 confidence. Never uses future audio.

Chain, identical to how training features were made:
  native-rate mic audio
  -> StreamingResampler   same filter as audio.io.resample (scipy resample_poly), but stateful
  -> int16 quantisation   as in the recorded / LibriSpeech files
  -> StreamingFrontend    frontend/streaming.py, the class used for training features
  -> global normaliser    constants stored in the checkpoint
  -> StreamingConvModel   re-runs Model A on a short window and releases an output frame only when
                          its whole receptive field has been heard (26 input frames = 260 ms ahead);
                          the live app WAITS for that audio instead of guessing it
  -> StreamingGreedy      CTC best path; a phoneme is released when its frame span ends
  -> dev-fitted temperature (calibration.json) -> level 0-7 (evaluation/calibration.py)

Pushing a whole recording in any chunk sizes and calling flush() reproduces the offline result
(resample -> extract_features -> model -> greedy_decode); see tests/test_live.py.
Model B (BiGRU) needs the whole window in both directions, so it cannot be streamed exactly; refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import torch
from scipy.signal import firwin

from ..audio.io import TARGET_SR, float_to_int16
from ..evaluation.calibration import apply_temperature, to_level
from ..frontend.config import FrontendConfig
from ..frontend.streaming import StreamingFrontend
from ..frontend.vad import VadSegmenter


class StreamingResampler:
    """Chunked equivalent of scipy.signal.resample_poly(x, up, down) (default Kaiser 5.0 window,
    zero padding). Outputs are released once every input sample they depend on has arrived."""

    def __init__(self, sr_in: int, sr_out: int = TARGET_SR):
        g = gcd(sr_in, sr_out)
        self.up, self.down = sr_out // g, sr_in // g
        self.n_in = 0           # input samples received
        self.m = 0              # next output index
        if self.up == self.down:
            return
        max_rate = max(self.up, self.down)
        half_len = 10 * max_rate
        h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self.up
        n_pre_pad = self.down - half_len % self.down
        h = np.concatenate([np.zeros(n_pre_pad), h])
        self.n_pre_remove = (half_len + n_pre_pad) // self.down
        self.K = -(-len(h) // self.up)                       # taps per polyphase branch
        self.hb = np.zeros((self.up, self.K))
        for r in range(self.up):
            taps = h[r:: self.up]
            self.hb[r, : len(taps)] = taps
        self.x = np.zeros(0)
        self.x_start = 0        # global index of self.x[0]

    def _outputs(self, m_end: int) -> np.ndarray:
        if m_end <= self.m:
            return np.zeros(0, np.float32)
        pos = (np.arange(self.m, m_end) + self.n_pre_remove) * self.down
        imax, r = pos // self.up, pos % self.up
        idx = imax[:, None] - np.arange(self.K)[None, :]     # global input indices
        ok = (idx >= 0) & (idx < self.n_in)
        vals = np.where(ok, self.x[np.clip(idx - self.x_start, 0, max(len(self.x) - 1, 0))], 0.0) \
            if len(self.x) else np.zeros(idx.shape)
        self.m = m_end
        return (self.hb[r] * vals).sum(1).astype(np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64).reshape(-1)
        self.n_in += len(x)
        if self.up == self.down:
            self.m += len(x)
            return x.astype(np.float32)
        self.x = np.concatenate([self.x, x])
        # output m needs input up to floor((m + pre) * down / up)
        m_end = max(self.m, (self.n_in * self.up - 1) // self.down - self.n_pre_remove + 1)
        y = self._outputs(m_end)
        keep_from = max(0, ((self.m + self.n_pre_remove) * self.down) // self.up - self.K + 1)
        if keep_from > self.x_start:
            self.x = self.x[keep_from - self.x_start:]
            self.x_start = keep_from
        return y

    def flush(self) -> np.ndarray:
        """End of stream: the rest, with zeros after the last sample (as resample_poly does)."""
        if self.up == self.down:
            return np.zeros(0, np.float32)
        return self._outputs(-(-self.n_in * self.up // self.down))


class StreamingConvModel:
    """Exact streaming for a fully convolutional model with symmetric 'same' padding (Model A)."""

    def __init__(self, model: torch.nn.Module):
        if isinstance(getattr(model, "gru", None), torch.nn.GRU):
            raise ValueError("Model B (BiGRU) cannot be streamed exactly; use Model A live")
        self.model = model.eval()
        self.stride = model.body.stem.stride
        rf = model.receptive_field()
        self.look = (rf - 1) // 2            # input frames needed on each side (26 for Model A)
        assert self.stride == 2 and self.look % 2 == 0, "window maths assumes stride 2, even lookahead"
        self.buf = np.zeros((0, 42), np.float32)
        self.buf_start = 0                   # global index (input frames) of buf[0]
        self.n = 0                           # input frames received
        self.j = 0                           # next output frame to release

    @torch.no_grad()
    def _release(self, j_end: int) -> np.ndarray:
        if j_end <= self.j:
            return np.zeros((0, 0), np.float32)
        s = max(0, 2 * self.j - self.look)   # even: window outputs line up with the global grid
        x = torch.from_numpy(np.ascontiguousarray(self.buf[s - self.buf_start:].T))[None]
        lp, _ = self.model(x)
        out = lp[0, self.j - s // 2: j_end - s // 2].float().numpy()
        self.j = j_end
        keep = max(0, 2 * self.j - self.look)
        if keep > self.buf_start:
            self.buf = self.buf[keep - self.buf_start:]
            self.buf_start = keep
        return out

    def push(self, feats: np.ndarray) -> np.ndarray:
        """feats: (n, 42) NORMALISED frames. Returns newly final (m, 41) log-probs (20 ms frames)."""
        if len(feats):
            self.buf = np.concatenate([self.buf, feats.astype(np.float32)])
            self.n += len(feats)
        # output j uses input frames 2j - look ... 2j + look
        return self._release(max(self.j, (self.n - 1 - self.look) // 2 + 1) if self.n > self.look else self.j)

    def flush(self) -> np.ndarray:
        return self._release(-(-self.n // 2))   # offline output length ceil(n / 2)


@dataclass
class Token:
    id: int
    label: str
    start: int          # output frame (20 ms) span [start, end)
    end: int
    conf: float
    level: int

    @property
    def start_s(self) -> float:
        return self.start * 0.02

    @property
    def end_s(self) -> float:
        return self.end * 0.02


class StreamingGreedy:
    """Greedy CTC over a frame stream: same tokens and spans as decoding.greedy.greedy_decode.
    Confidence = max posterior over the span after temperature T (as in calibration)."""

    def __init__(self, labels, blank_id: int = 0, temperature: float = 1.0):
        self.labels, self.blank, self.T = list(labels), blank_id, temperature
        self.prev = None
        self.cur = None        # [id, start, end, maxp]
        self.t = 0

    def _close(self, out):
        if self.cur is not None:
            k, a, b, p = self.cur
            out.append(Token(k, self.labels[k], a, b, p, int(to_level(p))))
            self.cur = None

    def push(self, lp: np.ndarray) -> list[Token]:
        out: list[Token] = []
        if len(lp) == 0:
            return out
        p = np.exp(apply_temperature(lp, self.T))
        for k, row in zip(p.argmax(-1), p):
            k = int(k)
            if k == self.prev:
                if self.cur is not None:
                    self.cur[2] = self.t + 1
                    self.cur[3] = max(self.cur[3], float(row[k]))
            else:
                self._close(out)
                if k != self.blank:
                    self.cur = [k, self.t, self.t + 1, float(row[k])]
            self.prev = k
            self.t += 1
        return out

    def flush(self) -> list[Token]:
        out: list[Token] = []
        self._close(out)
        return out


class LivePipeline:
    """Native-rate audio chunks in -> released Tokens (+ VAD events, in 10 ms frames) out."""

    def __init__(self, model, bundle: dict, sr_in: int, temperature: float = 1.0, blank_id: int = 0):
        self.resampler = StreamingResampler(sr_in)
        self.frontend = StreamingFrontend(FrontendConfig.from_dict(bundle["frontend"]))
        self.norm = bundle["normalizer"]
        self.net = StreamingConvModel(model)
        self.decoder = StreamingGreedy(bundle["labels"], blank_id, temperature)
        self.vad = VadSegmenter()
        self.n16 = 0             # 16 kHz samples fed to the front end
        self.n_feat = 0

    def _feed16(self, y: np.ndarray, final: bool = False) -> dict:
        self.n16 += len(y)
        fe = self.frontend.push(float_to_int16(y)) if len(y) else {"features": np.zeros((0, 42)), "vad": []}
        self.n_feat += len(fe["features"])
        ev = self.vad.push(fe["vad"])
        lp = self.net.push(self.norm(fe["features"]) if len(fe["features"]) else fe["features"])
        toks = self.decoder.push(lp)
        if final:
            toks += self.decoder.push(self.net.flush()) + self.decoder.flush()
        return {"audio16": y, "tokens": toks, "vad_events": ev}

    def push(self, audio: np.ndarray) -> dict:
        return self._feed16(self.resampler.push(audio))

    def flush(self) -> dict:
        return self._feed16(self.resampler.flush(), final=True)
