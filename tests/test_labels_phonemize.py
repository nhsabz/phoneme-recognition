import pytest

from phonrec.labels import PHONEMES_39, LabelMap, load_label_map
from phonrec.phonemize.lexicon import Lexicon, load_lexicon_file, strip_stress
from phonrec.phonemize.normalize_text import PAUSE_TOKEN, normalize_transcript, number_to_words
from phonrec.phonemize.phonemizer import Phonemizer


def test_label_map_frozen():
    lm = load_label_map()
    assert lm.num_classes == 41
    assert lm.blank_id == 0 and lm.labels[0] == "<blank>"
    assert lm.silence_id == 1 and lm.labels[1] == "SIL"
    assert lm.blank_id != lm.silence_id
    assert sorted(lm.labels[2:]) == sorted(PHONEMES_39)
    assert len(PHONEMES_39) == 39
    assert lm.decode(lm.encode(["SIL", "HH", "AH", "SIL"])) == ["SIL", "HH", "AH", "SIL"]


def test_blank_rejected_in_targets():
    with pytest.raises(ValueError):
        load_label_map().encode(["<blank>"])


def test_bad_label_maps_rejected():
    labels = list(load_label_map().labels)
    with pytest.raises(ValueError):
        LabelMap(tuple(labels[1:] + ["XX"]))
    swapped = labels.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        LabelMap(tuple(swapped))


def test_strip_stress():
    assert strip_stress(["AH0", "OW1", "IY2", "K"]) == ["AH", "OW", "IY", "K"]
    with pytest.raises(ValueError):
        strip_stress(["XX1"])


def test_normalize_punctuation_and_numbers():
    assert normalize_transcript("Hello, world!") == ["HELLO", "WORLD"]
    assert normalize_transcript("  don’t  'cat' ") == ["DON'T", "CAT"]
    assert normalize_transcript("the cat-sat") == ["THE", "CAT", "SAT"]
    assert normalize_transcript("42") == ["FORTY", "TWO"]
    assert number_to_words(1905) == "ONE THOUSAND NINE HUNDRED FIVE"
    assert normalize_transcript("hello <pause> world") == ["HELLO", PAUSE_TOKEN, "WORLD"]
    assert normalize_transcript("...!!") == []


def test_lexicon_first_variant_and_comments(cmu_path):
    lex = load_lexicon_file(cmu_path)
    assert lex["HELLO"] == ["HH", "AH", "L", "OW"]   # (2) variant ignored
    assert "HELLO(2)" not in lex


def test_phonemize_common_words(cmu_path):
    ph = Phonemizer(Lexicon.from_files(cmu_path), load_label_map())
    r = ph.phonemize("Hello, world.")
    assert r.phones == ["SIL", "HH", "AH", "L", "OW", "W", "ER", "L", "D", "SIL"]
    assert r.oov_words == []
    ids = ph.encode("the cat")
    lm = load_label_map()
    assert ids[0] == lm.silence_id and ids[-1] == lm.silence_id
    assert 0 not in ids


def test_pause_becomes_silence_and_merges(cmu_path):
    ph = Phonemizer(Lexicon.from_files(cmu_path), load_label_map())
    r = ph.phonemize("<pause> cat <pause> <pause> sat <pause>")
    assert r.phones == ["SIL", "K", "AE", "T", "SIL", "S", "AE", "T", "SIL"]


def test_unknown_word_reported(cmu_path):
    lex = Lexicon.from_files(cmu_path)
    ph = Phonemizer(lex, load_label_map())
    r = ph.phonemize("hello zyxqu")
    assert r.phones is None and r.oov_words == ["ZYXQU"]
    assert lex.oov["ZYXQU"] == 1
    assert ph.encode("hello zyxqu") is None


def test_override_wins(cmu_path, tmp_path):
    ov = tmp_path / "overrides.txt"
    ov.write_text("ZYXQU  Z IH1 K S\nCAT  K AA1 T\n")
    ph = Phonemizer(Lexicon.from_files(cmu_path, ov), load_label_map())
    assert ph.phonemize("zyxqu cat").phones == ["SIL", "Z", "IH", "K", "S", "K", "AA", "T", "SIL"]


def test_empty_transcript_is_rejected(cmu_path):
    ph = Phonemizer(Lexicon.from_files(cmu_path), load_label_map())
    assert ph.phonemize("!!!").phones is None
