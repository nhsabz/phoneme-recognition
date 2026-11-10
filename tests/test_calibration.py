import numpy as np

from phonrec.decoding.greedy import greedy_decode
from phonrec.evaluation.calibration import (apply_temperature, collect_tokens, ece, fit_temperature,
                                            level_table, to_level, token_confidences)
from phonrec.evaluation.per import align, align_ops, hyp_correct


def test_align_ops_matches_stats():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ref = list(rng.integers(2, 8, rng.integers(0, 12)))
        hyp = list(rng.integers(2, 8, rng.integers(0, 12)))
        st = align(ref, hyp)
        ops = [o for o, _, _ in align_ops(ref, hyp)]
        assert ops.count("sub") == st.sub and ops.count("del") == st.dele and ops.count("ins") == st.ins
        ok = hyp_correct(ref, hyp)
        assert len(ok) == len(hyp)
        assert sum(ok) == len(ref) - st.sub - st.dele


def test_hyp_correct_simple():
    assert hyp_correct([2, 3, 4], [2, 9, 4, 5]) == [True, False, True, False]


def test_temperature_keeps_argmax_and_normalises():
    lp = np.log(np.random.default_rng(1).dirichlet(np.ones(41), size=50)).astype(np.float32)
    for T in (0.5, 1.0, 3.0):
        q = apply_temperature(lp, T)
        assert np.allclose(np.exp(q).sum(-1), 1, atol=1e-5)
        assert (q.argmax(-1) == lp.argmax(-1)).all()
    assert np.allclose(apply_temperature(lp, 1.0), lp, atol=1e-5)


def test_token_confidence_is_max_over_span():
    p = np.full((6, 41), 1e-4)
    p[:, 0] = 1.0                       # blank everywhere ...
    p[1, 5], p[2, 5] = 3.0, 6.0         # ... except a 2-frame token 5
    p /= p.sum(1, keepdims=True)
    ids, confs, spans = token_confidences(np.log(p))
    assert ids == [5] and spans == [(1, 3)]
    assert np.isclose(confs[0], p[1:3, 5].max())


def test_levels():
    assert list(to_level([0.0, 0.124, 0.125, 0.5, 0.99, 1.0])) == [0, 0, 1, 4, 7, 7]
    tab = level_table(np.array([0.05, 0.9, 0.95]), np.array([False, True, True]))
    assert tab[0]["n"] == 1 and tab[7]["n"] == 2 and tab[7]["accuracy"] == 1.0


def _synthetic(T0, n=4000, seed=0):
    """One-frame tokens whose true P(correct) is softmax(z / T0)[k] while the model reports softmax(z)."""
    rng = np.random.default_rng(seed)
    lps, refs = {}, {}
    for u in range(n):
        z = rng.normal(0, 1, 41)
        k = int(rng.integers(2, 41))
        z[k] = z.max() + rng.uniform(0.5, 8)
        lp = np.stack([np.log(np.eye(41)[0] * 0.98 + 0.02 / 41), z - np.log(np.exp(z).sum())])
        p_true = np.exp(apply_temperature(lp[1:], T0))[0, k]
        refs[f"{u}-0-0"] = [k] if rng.random() < p_true else [2 if k != 2 else 3]
        lps[f"{u}-0-0"] = lp.astype(np.float32)
    return lps, refs


def test_fit_temperature_recovers_overconfidence():
    lps, refs = _synthetic(T0=2.0)
    ts = collect_tokens(lps, refs)
    assert len(ts.cls) == len(refs)
    assert all(greedy_decode(lps[u]) == [k] for u, k in zip(ts.uid[:20], ts.cls[:20]))
    T = fit_temperature(ts)
    assert 1.6 < T < 2.5
    assert ece(ts.confidences(T), ts.correct) < ece(ts.confidences(1.0), ts.correct)


def test_subset_roundtrip():
    lps, refs = _synthetic(T0=1.0, n=50)
    ts = collect_tokens(lps, refs)
    m = np.arange(len(ts.cls)) % 2 == 0
    sub = ts.subset(m)
    assert np.allclose(sub.confidences(1.3), ts.confidences(1.3)[m])
