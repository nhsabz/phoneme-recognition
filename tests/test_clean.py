import numpy as np

from phonrec.audio.clean import trim_recording

SR = 16000


def _take(click=0, speech_to_end=False, seed=0):
    rng = np.random.default_rng(seed)
    room = lambda n: 1e-3 * rng.standard_normal(n)  # noqa: E731  (~ -60 dBFS)
    t = np.arange(SR) / SR
    speech = 0.1 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    parts = [np.zeros(int(0.25 * SR)), room(int(0.3 * SR)), speech]
    if not speech_to_end:
        parts.append(room(int(0.4 * SR)))
    if click:
        parts.append(0.05 * rng.standard_normal(click * SR // 1000))
    return np.concatenate(parts).astype(np.float32)


def test_removes_leading_digital_silence_and_click():
    x = _take(click=40)
    y, info = trim_recording(x)
    assert info == {"lead_ms": 250, "tail_ms": 40}
    assert len(y) == len(x) - int(0.25 * SR) - 40 * SR // 1000


def test_no_click_keeps_end():
    x = _take(click=0)
    y, info = trim_recording(x)
    assert info["tail_ms"] == 0 and len(y) == len(x) - int(0.25 * SR)


def test_speech_running_to_the_end_is_kept():
    x = _take(speech_to_end=True)
    _, info = trim_recording(x)
    assert info["tail_ms"] == 0


def test_long_burst_is_not_a_click():
    _, info = trim_recording(_take(click=400))
    assert info["tail_ms"] == 0
