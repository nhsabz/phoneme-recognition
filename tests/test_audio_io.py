import numpy as np

from phonrec.audio.io import check_wav_spec, level_report, read_wav, resample, write_wav


def test_wav_roundtrip_and_spec(tmp_path):
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    p = tmp_path / "a.wav"
    write_wav(p, x)
    info = check_wav_spec(p)
    assert info == {"sr": 16000, "channels": 1, "bits": 16, "seconds": 1.0, "ok": True}
    y, sr = read_wav(p)
    assert sr == 16000 and np.abs(y - x).max() < 1e-4


def test_resample_48k_to_16k_keeps_tone():
    sr = 48000
    x = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
    y = resample(x, sr)
    assert len(y) == 16000
    spec = np.abs(np.fft.rfft(y[1000:15000] * np.hanning(14000)))
    peak_hz = spec.argmax() * 16000 / 14000
    assert abs(peak_hz - 1000) < 3


def test_level_warnings():
    assert level_report(np.ones(100))["warnings"]
    assert level_report(1e-4 * np.ones(100))["warnings"]
    assert not level_report(0.1 * np.sin(np.arange(1000)))["warnings"]
