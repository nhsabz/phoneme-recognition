"""Front-end configuration. Defaults follow the spec; (A) marks assumptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FrontendConfig:
    sample_rate: int = 16000
    win_length: int = 400          # 25 ms
    hop_length: int = 160          # 10 ms
    n_fft: int = 512
    n_mels: int = 40
    fmin: float = 80.0
    fmax: float = 7600.0
    input_scale: float = 32768.0   # (A) float audio in [-1,1] is scaled to int16 range

    # cleanup
    dc_pole: float = 0.995         # (A) DC blocker pole (~13 Hz)
    hpf_hz: float = 80.0
    hpf_order: int = 2             # (A) Butterworth order

    # minimum-statistics noise tracker  (A) Martin-style, simplified
    ms_alpha: float = 0.85         # power smoothing
    ms_sub_frames: int = 25        # V: sub-window length (250 ms)
    ms_num_sub: int = 6            # U: number of sub-windows -> 1.5 s search window
    ms_bias: float = 1.5           # bias compensation of the minimum

    # Wiener (decision-directed)
    wiener: bool = True            # False = bypass (gain 1); always A/B tested
    dd_alpha: float = 0.98
    gain_floor: float = 0.1        # amplitude floor (-20 dB)

    # features
    feature: str = "pcen"          # "pcen" (preferred) or "logmel"
    pcen_s: float = 0.025
    pcen_alpha: float = 0.98
    pcen_delta: float = 2.0
    pcen_r: float = 0.5
    pcen_eps: float = 1e-6

    # pitch / voicing  (A) normalised autocorrelation on past-only 40 ms window
    pitch_win: int = 640
    f0_min: float = 60.0
    f0_max: float = 400.0
    voicing_threshold: float = 0.5
    f0_ref_hz: float = 150.0

    # soft VAD  (A)
    vad_lo_hz: float = 300.0
    vad_hi_hz: float = 4000.0
    vad_snr_db: float = 6.0
    vad_slope_db: float = 2.0
    vad_attack: float = 0.5
    vad_release: float = 0.05

    @property
    def num_channels(self) -> int:
        return self.n_mels + 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrontendConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
