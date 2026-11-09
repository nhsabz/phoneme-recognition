"""PyTorch dataset, length-bucketed batch sampler, SpecAugment and collate."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..data.store import FeatureStore
from ..frontend.normalize import FeatureNormalizer


class PhonemeDataset(Dataset):
    """Returns (features (T, 42) normalised, label ids, uid). Picks a random variant when training."""

    def __init__(self, store: FeatureStore, normalizer: FeatureNormalizer, train: bool, seed: int = 0):
        self.store, self.norm, self.train = store, normalizer, train
        self.ids = store.ids
        # python's global `random` is re-seeded per DataLoader worker by PyTorch,
        # so variant choices differ across workers but are reproducible per run

    def __len__(self):
        return len(self.ids)

    def lengths(self) -> list[int]:
        return [self.store.n_frames(u) for u in self.ids]

    def __getitem__(self, i):
        uid = self.ids[i]
        v = random.choice(self.store.variants) if self.train else self.store.variants[0]
        f = self.norm(self.store.get(uid, v))
        return torch.from_numpy(f), torch.tensor(self.store.labels(uid), dtype=torch.long), uid


class BucketBatchSampler(Sampler):
    """Groups utterances of similar length (less padding). Batch order is shuffled
    each epoch with a seed derived from (seed, epoch) -> reproducible and resumable."""

    def __init__(self, lengths, batch_size: int, shuffle: bool, seed: int = 0, jitter: float = 0.1):
        self.lengths = np.asarray(lengths)
        self.bs, self.shuffle, self.seed, self.jitter = batch_size, shuffle, seed, jitter
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def _batches(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        key = self.lengths * (1 + self.jitter * rng.random(len(self.lengths))) if self.shuffle else self.lengths
        order = np.argsort(key, kind="stable")
        batches = [order[i: i + self.bs].tolist() for i in range(0, len(order), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def spec_augment(x: torch.Tensor, n_freq=2, max_f=8, n_time=2, max_t=40, n_mel=40, gen=None):
    """x: (T, 42). Frequency masks only on the 40 mel channels; time masks on all channels.
    Masked values are set to 0 (= the training mean after normalisation)."""
    T = x.shape[0]
    x = x.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, max_f + 1, (1,), generator=gen))
        f0 = int(torch.randint(0, max(1, n_mel - f + 1), (1,), generator=gen))
        x[:, f0: f0 + f] = 0.0
    for _ in range(n_time):
        t = int(torch.randint(0, min(max_t, max(T // 5, 1)) + 1, (1,), generator=gen))
        t0 = int(torch.randint(0, max(1, T - t + 1), (1,), generator=gen))
        x[t0: t0 + t, :] = 0.0
    return x


def make_collate(augment: bool, sa_cfg: dict | None = None):
    sa_cfg = sa_cfg or {}

    def collate(batch):
        feats, labels, uids = zip(*batch)
        if augment:
            feats = [spec_augment(f, **sa_cfg) for f in feats]
        lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        T = int(lens.max())
        x = torch.zeros(len(feats), feats[0].shape[1], T)
        for i, f in enumerate(feats):
            x[i, :, : f.shape[0]] = f.T
        tl = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
        return x, lens, torch.cat(labels), tl, list(uids)

    return collate
