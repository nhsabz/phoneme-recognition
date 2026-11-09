"""Atomic, self-contained checkpoints.

A checkpoint holds everything the PC live app needs: model weights, model
config, frontend config, label list and normaliser constants -- plus the
optimiser / scaler / scheduler / RNG state for resuming on Colab.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: str | Path, state: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    # weights_only=False: our own checkpoints contain numpy RNG state / configs
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_bundle(path: str | Path, device="cpu"):
    """For evaluation / live use: returns (model, bundle dict)."""
    from ..frontend.normalize import FeatureNormalizer
    from ..models.models import build_model
    ck = load_checkpoint(path, device)
    model = build_model(ck["model_name"], **ck.get("model_kwargs", {}))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    norm = FeatureNormalizer(ck["norm_mean"], ck["norm_std"])
    return model, {**ck, "normalizer": norm}
