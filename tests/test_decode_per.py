import numpy as np

from phonrec.decoding.greedy import collapse, greedy_decode, strip_silence
from phonrec.evaluation.per import align, corpus_per


def onehot_path(path, C=41):
    lp = np.full((len(path), C), -10.0)
    lp[np.arange(len(path)), path] = 0.0
    return lp


def test_collapse_rules():
    # blank=0, SIL=1, T=32 (any id works)
    assert collapse([0, 5, 5, 0, 5, 7, 7, 0]) == [5, 5, 7]    # blank separates repeats
    assert collapse([5, 5, 5]) == [5]                         # no blank -> one token
    assert collapse([0, 0, 0]) == []
    assert collapse([1, 1, 0, 9, 0, 1]) == [1, 9, 1]


def test_greedy_decode_and_spans():
    lp = onehot_path([0, 1, 1, 0, 8, 8, 0, 8, 0, 1])
    ids, spans = greedy_decode(lp, return_frames=True)
    assert ids == [1, 8, 8, 1]
    assert spans == [(1, 3), (4, 6), (7, 8), (9, 10)]
    assert strip_silence(ids) == [8, 8]


def test_per_counts():
    s = align(list("abcd"), list("abcd"))
    assert (s.sub, s.dele, s.ins, s.per) == (0, 0, 0, 0.0)
    s = align(list("abcd"), list("axd"))          # b->x sub, c deleted
    assert (s.sub, s.dele, s.ins) == (1, 1, 0) and s.per == 0.5
    s = align(list("ab"), list("aab"))
    assert (s.sub, s.dele, s.ins) == (0, 0, 1)
    s = align([], list("ab"))
    assert s.ins == 2
    tot = corpus_per([list("abcd"), list("ab")], [list("axd"), list("ab")])
    assert tot.n_ref == 6 and tot.errors == 2
    # 'abcd'->'axd' has two optimal alignments (b->x + del c, or del b + c->x)
    assert (tot.pairs[("b", "x")], tot.pairs[("c", "")]) in {(1, 1), (0, 0)}
    assert tot.sub == 1 and tot.dele == 1
