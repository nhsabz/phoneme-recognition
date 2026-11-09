"""Model A (small, Zero-friendly) and Model B (stronger, PC only).

Both: input (B, 42, T) at 10 ms; output (B, T', 41) log-probabilities at 20 ms,
T' = ceil(T / 2). Trained with CTC (blank = 0).

Model A widths: the spec says "widths 128 to 192" without the exact steps.
Chosen (decision D1): blocks 128->128, 128->128, 128->192, 192->192
(residual on blocks 1, 2, 4). See tests/test_models.py for the measured count.
"""
from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvStem, DSBlock, output_lengths, receptive_field

DILATIONS = (1, 1, 2, 2)


class _ConvBody(nn.Module):
    def __init__(self, c_in, stem_ch, widths, kernel=5, dropout=0.1):
        super().__init__()
        self.stem = ConvStem(c_in, stem_ch, kernel=5, stride=2)
        chans = [stem_ch, *widths]
        self.blocks = nn.Sequential(*[
            DSBlock(chans[i], chans[i + 1], kernel, DILATIONS[i], dropout) for i in range(len(widths))
        ])
        self.out_ch = chans[-1]

    def forward(self, x):
        return self.blocks(self.stem(x))

    def receptive_field(self) -> int:
        layers = [(self.stem.kernel, self.stem.stride, 1)]
        for b in self.blocks:
            layers.append((b.kernel, 1, b.dilation))
        return receptive_field(layers)


class ModelA(nn.Module):
    def __init__(self, in_ch=42, num_classes=41, widths=(128, 128, 192, 192), dropout=0.1):
        super().__init__()
        self.body = _ConvBody(in_ch, 128, widths, dropout=dropout)
        self.head = nn.Conv1d(self.body.out_ch, num_classes, 1)

    def forward(self, x, lengths=None):
        y = self.head(self.body(x)).transpose(1, 2)          # (B, T', C)
        lp = y.log_softmax(-1)
        return lp, (None if lengths is None else output_lengths(lengths))

    def logits(self, x):
        return self.head(self.body(x)).transpose(1, 2)

    def receptive_field(self) -> int:
        return self.body.receptive_field()


class ModelB(nn.Module):
    def __init__(self, in_ch=42, num_classes=41, width=256, gru_hidden=256, gru_layers=2, dropout=0.1):
        super().__init__()
        self.body = _ConvBody(in_ch, width, (width,) * 4, dropout=dropout)
        self.gru = nn.GRU(width, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.head = nn.Linear(2 * gru_hidden, num_classes)

    def logits(self, x, lengths=None):
        h = self.body(x).transpose(1, 2)                      # (B, T', 256)
        if lengths is not None:
            ol = output_lengths(lengths).clamp(max=h.shape[1])
            packed = nn.utils.rnn.pack_padded_sequence(h, ol.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        else:
            h, _ = self.gru(h)
        return self.head(h)

    def forward(self, x, lengths=None):
        lp = self.logits(x, lengths).log_softmax(-1)
        return lp, (None if lengths is None else output_lengths(lengths))

    def receptive_field(self) -> int:
        return self.body.receptive_field()  # conv part only; BiGRU adds whole-utterance context


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name in ("a", "model_a"):
        return ModelA(**kw)
    if name in ("b", "model_b"):
        return ModelB(**kw)
    raise ValueError(name)
