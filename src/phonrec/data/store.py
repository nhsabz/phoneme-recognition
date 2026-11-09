"""Sharded on-disk feature store.

Layout:  <root>/<split>/<variant>/shard_00000.npy   float16 (n_frames_total, 42)
                                   shard_00000.json  [{"id", "offset", "n_frames", "labels", ...meta}]
A shard is written only when complete (tmp file + rename), so an interrupted
build can resume by skipping shards that already exist.
Variants: "clean" (no augmentation) and "aug<k>" (gain + added noise, built offline).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class ShardWriter:
    def __init__(self, directory: str | Path, shard_index: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k = shard_index
        self.feats: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.offset = 0

    def add(self, entry: dict, feats: np.ndarray):
        e = dict(entry)
        e["offset"] = self.offset
        e["n_frames"] = int(len(feats))
        self.entries.append(e)
        self.feats.append(feats.astype(np.float16))
        self.offset += len(feats)

    def close(self):
        npy = self.dir / f"shard_{self.k:05d}.npy"
        js = self.dir / f"shard_{self.k:05d}.json"
        arr = np.concatenate(self.feats) if self.feats else np.zeros((0, 42), np.float16)
        tmp = npy.with_name(npy.stem + ".tmp.npy")  # np.save keeps the .npy suffix
        np.save(tmp, arr)
        os.replace(tmp, npy)
        tmpj = js.with_suffix(".json.tmp")
        tmpj.write_text(json.dumps(self.entries))
        os.replace(tmpj, js)          # json last: its presence marks the shard complete


def shard_done(directory: str | Path, k: int) -> bool:
    return (Path(directory) / f"shard_{k:05d}.json").exists()


class FeatureStore:
    """Read-only view of one split; groups variants of the same utterance."""

    def __init__(self, root: str | Path, split: str, variants=("clean",), mmap: bool = True):
        self.root = Path(root)
        self.split = split
        self.variants = tuple(variants)
        self._arrays: dict[tuple[str, int], np.ndarray] = {}
        self.items: dict[str, dict] = {}       # id -> meta incl. labels
        self.loc: dict[str, dict[str, tuple[int, int, int]]] = {}   # id -> variant -> (shard, off, n)
        for v in self.variants:
            d = self.root / split / v
            if not d.exists():
                raise FileNotFoundError(f"missing feature dir {d}")
            for js in sorted(d.glob("shard_*.json")):
                k = int(js.stem.split("_")[1])
                self._arrays[(v, k)] = np.load(js.with_suffix(".npy"), mmap_mode="r" if mmap else None)
                for e in json.loads(js.read_text()):
                    uid = e["id"]
                    self.items.setdefault(uid, {kk: vv for kk, vv in e.items() if kk not in ("offset", "n_frames")})
                    self.loc.setdefault(uid, {})[v] = (k, e["offset"], e["n_frames"])
        # keep only utterances present in every requested variant
        self.ids = sorted(u for u, l in self.loc.items() if all(v in l for v in self.variants))

    def __len__(self):
        return len(self.ids)

    def n_frames(self, uid: str) -> int:
        return self.loc[uid][self.variants[0]][2]

    def get(self, uid: str, variant: str | None = None) -> np.ndarray:
        v = variant or self.variants[0]
        k, off, n = self.loc[uid][v]
        return np.asarray(self._arrays[(v, k)][off: off + n], dtype=np.float32)

    def labels(self, uid: str) -> list[int]:
        return self.items[uid]["labels"]

    def meta(self, uid: str) -> dict:
        return self.items[uid]
