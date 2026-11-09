"""Live (chunked) features must equal offline (whole-utterance) features."""
import numpy as np
import pytest

from phonrec.frontend.config import FrontendConfig
from phonrec.frontend.streaming import StreamingFrontend, extract_features

SR = 16000


def chunked(x, sizes, cfg):
    fe = StreamingFrontend(cfg)
    feats, vads, i = [], [], 0
    k = 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out = fe.push(x[i:i + n])
        feats.append(out["features"])
        vads.append(out["vad"])
        i += n
        k += 1
    return np.concatenate(feats), np.concatenate(vads)


@pytest.mark.parametrize("wiener", [True, False])
@pytest.mark.parametrize("sizes", [[160], [1], [37, 512, 3, 1600], [4000]])
def test_chunking_invariance(wiener, sizes):
    rng = np.random.default_rng(1)
    t = np.arange(int(1.3 * SR)) / SR
    x = 0.02 * rng.standard_normal(len(t)) + 0.2 * np.sin(2 * np.pi * 180 * t) * (t > 0.5)
    if sizes == [1]:
        x = x[:3000]  # keep the per-sample case fast
    cfg = FrontendConfig(wiener=wiener)
    ref = extract_features(x, cfg)
    f, v = chunked(x, sizes, cfg)
    assert f.shape == ref["features"].shape
    np.testing.assert_allclose(f, ref["features"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(v, ref["vad"], rtol=1e-5, atol=1e-5)


def test_no_future_dependence():
    """Frame t must not change when later audio changes."""
    rng = np.random.default_rng(2)
    x = 0.05 * rng.standard_normal(SR)
    y = x.copy()
    y[8000:] = rng.standard_normal(SR - 8000)          # different future
    fx = extract_features(x)["features"]
    fy = extract_features(y)["features"]
    last_safe = (8000 - 400) // 160                     # frames ending before sample 8000
    np.testing.assert_array_equal(fx[: last_safe + 1], fy[: last_safe + 1])
    assert not np.allclose(fx[last_safe + 2:], fy[last_safe + 2:])
