import numpy as np
import pytest

from phonrec.frontend.config import FrontendConfig
from phonrec.frontend.mel import mel_filterbank
from phonrec.frontend.pitch import estimate_f0
from phonrec.frontend.streaming import StreamingFrontend, extract_features, num_frames
from phonrec.frontend.vad import segments_from_probs

SR = 16000
rng = np.random.default_rng(0)


def voiced(f0, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return amp * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))


@pytest.mark.parametrize("wiener", [True, False])
@pytest.mark.parametrize("feature", ["pcen", "logmel"])
def test_shapes_and_finite(wiener, feature):
    cfg = FrontendConfig(wiener=wiener, feature=feature)
    x = 0.01 * rng.standard_normal(SR)
    out = extract_features(x, cfg)
    assert out["features"].shape == (num_frames(SR), 42) == (98, 42)
    assert out["vad"].shape == (98,)
    assert np.isfinite(out["features"]).all()
    assert out["features"].dtype == np.float32


def test_short_and_silent_input():
    assert extract_features(np.zeros(399))["features"].shape == (0, 42)
    f = extract_features(np.zeros(SR))["features"]
    assert np.isfinite(f).all()
    assert (f[:, 41] == 0).all() and (f[:, 40] == 0).all()   # digital silence: unvoiced


def test_int16_input_matches_float():
    x = (0.1 * rng.standard_normal(8000) * 32767).astype(np.int16)
    a = extract_features(x)["features"]
    b = extract_features(x.astype(np.float64) / 32768.0)["features"]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_mel_bank_covers_range_and_no_empty_band():
    fb = mel_filterbank(SR, 512, 40, 80, 7600)
    assert fb.shape == (40, 257)
    assert (fb.sum(1) > 0).all()
    freqs = np.fft.rfftfreq(512, 1 / SR)
    assert fb[:, freqs < 80].sum() == 0 and fb[:, freqs > 7600].sum() == 0


def test_highpass_removes_dc_and_low_hum():
    t = np.arange(2 * SR) / SR
    fe = StreamingFrontend(FrontendConfig())
    fe.push(np.zeros(1))
    y_in = 0.5 + 0.3 * np.sin(2 * np.pi * 30 * t)          # DC + 30 Hz hum
    from scipy.signal import sosfilt
    y = sosfilt(fe.sos, y_in)
    assert abs(y[SR:].mean()) < 1e-3                        # DC removed
    assert np.abs(y[SR:]).max() < 0.15 * 0.3                # 2nd-order 80 Hz HPF: ~18 dB at 30 Hz
    y1k = sosfilt(fe.sos, np.sin(2 * np.pi * 1000 * t))
    assert 0.95 < np.abs(y1k[SR:]).max() < 1.05            # passband untouched


@pytest.mark.parametrize("f0", [100.0, 150.0, 220.0, 300.0])
def test_pitch_on_harmonic_signal(f0):
    out = extract_features(voiced(f0, 0.5))["features"]
    est = 150.0 * 2.0 ** out[10:, 40]
    assert (out[10:, 41] > 0.8).all()
    assert np.median(np.abs(est / f0 - 1)) < 0.02


def test_noise_is_unvoiced():
    out = extract_features(0.1 * rng.standard_normal(SR))["features"]
    assert np.median(out[:, 41]) < 0.5
    assert np.mean(out[:, 40] == 0) > 0.7


def test_wiener_suppresses_stationary_noise():
    noise = 0.02 * rng.standard_normal(3 * SR)
    sig = noise.copy()
    sig[2 * SR: int(2.5 * SR)] += voiced(150, 0.5)
    on = StreamingFrontend(FrontendConfig(wiener=True, feature="logmel")).push(sig)["features"][:, :40]
    off = StreamingFrontend(FrontendConfig(wiener=False, feature="logmel")).push(sig)["features"][:, :40]
    noise_frames = slice(160, 190)   # 1.6-1.9 s: tracker settled, noise only
    speech_frames = slice(210, 240)
    assert on[noise_frames].mean() < off[noise_frames].mean() - 1.0     # noise reduced
    # speech kept: suppression on speech much smaller than on noise
    assert (off[speech_frames].mean() - on[speech_frames].mean()) < \
           (off[noise_frames].mean() - on[noise_frames].mean())


def test_vad_detects_speech_and_preroll():
    sig = 0.005 * rng.standard_normal(4 * SR)
    sig[2 * SR: 3 * SR] += voiced(150, 1.0)
    vad = extract_features(sig)["vad"]
    assert vad[150:190].mean() < 0.3
    assert vad[205:295].mean() > 0.7
    segs = segments_from_probs(vad, pre_roll=20, hangover=30)
    assert len(segs) == 1
    s, e = segs[0]
    assert s <= 200 - 15           # opens >= ~150 ms before onset (pre-roll)
    assert e >= 300                # does not cut the end
