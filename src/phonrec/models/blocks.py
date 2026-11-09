"""Shared building blocks for Model A and Model B (PyTorch)."""
from __future__ import annotations

import torch
from torch import nn


class ConvStem(nn.Module):
    """Conv 1x5, stride 2: 42 -> C. Halves the frame rate (10 ms -> 20 ms)."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        self.kernel, self.stride, self.dilation = kernel, stride, 1
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSBlock(nn.Module):
    """Depthwise-separable block: depthwise k5 (dilated) -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU.

    'same' padding (symmetric context). Residual connection when c_in == c_out.
    BatchNorm and ReLU fold into the convolutions for later INT8 export.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.kernel, self.stride, self.dilation = kernel, 1, dilation
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(c_in, c_in, kernel, padding=pad, dilation=dilation, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm1d(c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.residual = c_in == c_out

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.drop(self.act(self.bn2(self.pw(y))))
        return x + y if self.residual else y


def receptive_field(layers) -> int:
    """Receptive field in INPUT frames of a stack of (kernel, stride, dilation) layers."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf


def output_lengths(in_lengths: torch.Tensor, kernel: int = 5, stride: int = 2) -> torch.Tensor:
    pad = kernel // 2
    return torch.div(in_lengths + 2 * pad - kernel, stride, rounding_mode="floor") + 1


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
