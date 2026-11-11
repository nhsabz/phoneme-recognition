"""Live path == offline path (chunking, resampling, model windowing, greedy), and no future audio."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phonrec.audio.io import float_to_int16, resample  # noqa: E402
from phonrec.decoding.greedy import greedy_decode  # noqa: E402
from phonrec.evaluation.calibration import apply_temperature  # noqa: E402
from phonrec.frontend.config import FrontendConfig  # noqa: E402
from phonrec.frontend.normalize import FeatureNormalizer  # noqa: E402
from phonrec.frontend.streaming import extract_features  # noqa: E402
from phonrec.labels import load_label_map  # noqa: E402
from phonrec.models.models import ModelA, ModelB  # noqa: E402
from phonrec.streaming.live import (LivePipeline, StreamingConvModel, StreamingGreedy,  # noqa: E402
                                    StreamingResampler)


def _chunks(x, rng, lo=1, hi=3000):
    i = 0
    while i < len(x):
        n = int(rng.integers(lo, hi))
        yield x[i: i + n]
        i += n


def _model(seed=0):
    torch.manual_seed(seed)
    m = ModelA().eval()
    with torch.no_grad():               # non-trivial BN statistics
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    return m


@pytest.mark.parametrize("sr", [44100, 48000, 16000, 22050])
def test_streaming_resampler_matches_resample_poly(sr):
    rng = np.random.default_rng(sr)
    x = rng.standard_normal(int(sr * 0.7)).astype(np.float32) * 0.3
    rs = StreamingResampler(sr)
    y = np.concatenate([rs.push(c) for c in _chunks(x, rng)] + [rs.flush()])
    ref = resample(x, sr)
    assert len(y) == len(ref)
    assert np.allclose(y, ref, atol=1e-5)


def test_resampler_no_future_dependence():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(44100).astype(np.float32)
    a, b = StreamingResampler(44100), StreamingResampler(44100)
    ya = a.push(x[:30000])
    x2 = x.copy()
    x2[30000:] = rng.standard_normal(44100 - 30000)
    yb = b.push(x2[:30000])
    assert len(ya) > 0 and np.array_equal(ya, yb)


def test_streaming_model_matches_offline():
    rng = np.random.default_rng(0)
    m = _model()
    f = rng.standard_normal((357, 42)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(f.T.copy())[None])[0][0].numpy()
    sm = StreamingConvModel(m)
    parts = [sm.push(c) for c in _chunks(f, rng, 1, 40)]
    got = np.concatenate([p for p in parts if p.size] + [sm.flush()])
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4)


def test_streaming_model_waits_for_lookahead():
    m = _model()
    sm = StreamingConvModel(m)
    f = np.random.default_rng(2).standard_normal((100, 42)).astype(np.float32)
    assert sm.push(f[:26]).size == 0                 # 26 frames: nothing final yet
    assert len(sm.push(f[26:27])) == 1               # 27th frame completes output 0's context
    assert len(sm.push(f[27:100])) == (100 - 27) // 2 + 1 - 1


def test_model_b_refused():
    with pytest.raises(ValueError):
        StreamingConvModel(ModelB())


def test_streaming_greedy_matches_greedy_decode():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, 400)
    a[rng.random(400) < 0.5] = 0
    lp = np.log(np.full((400, 41), 1e-3) + 3.0 * np.eye(41)[a] + rng.random((400, 41)) * 1e-3)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    ids, spans = greedy_decode(lp, return_frames=True)
    dec = StreamingGreedy(load_label_map().labels, temperature=1.7)
    toks = [t for c in _chunks(lp, rng, 1, 30) for t in dec.push(c)] + dec.flush()
    assert [t.id for t in toks] == ids and [(t.start, t.end) for t in toks] == spans
    p = np.exp(apply_temperature(lp, 1.7))
    assert np.allclose([t.conf for t in toks], [p[s:e, k].max() for k, (s, e) in zip(ids, spans)])


def test_live_pipeline_matches_offline_path():
    rng = np.random.default_rng(4)
    sr = 44100
    n = int(sr * 0.08)                  # 80 ms segments of varied tone + noise -> varied model output
    tt = np.arange(n) / sr
    x = np.concatenate([rng.uniform(0.01, 0.4) * np.sin(2 * np.pi * rng.uniform(100, 3000) * tt)
                        + rng.uniform(0, 0.1) * rng.standard_normal(n) for _ in range(20)]).astype(np.float32)
    m = _model(1)
    cfg = FrontendConfig()
    raw = extract_features(float_to_int16(resample(x, sr)), cfg)["features"]
    norm = FeatureNormalizer(raw.mean(0), raw.std(0) + 1e-6)     # fixed constants, same for both paths
    bundle = {"frontend": cfg.to_dict(), "normalizer": norm, "labels": load_label_map().labels}
    # offline: what training / evaluate.py do
    f = norm(raw)
    with torch.no_grad():
        lp = m(torch.from_numpy(f.T.copy())[None])[0][0].numpy()
    ids, spans = greedy_decode(lp, return_frames=True)
    assert len(ids) > 5                                  # non-trivial comparison
    # live
    pipe = LivePipeline(m, bundle, sr, temperature=1.0)
    toks = [tk for c in _chunks(x, rng, 200, 2205) for tk in pipe.push(c)["tokens"]] + pipe.flush()["tokens"]
    assert [tk.id for tk in toks] == ids and [(tk.start, tk.end) for tk in toks] == spans
    assert pipe.n_feat == len(f)
