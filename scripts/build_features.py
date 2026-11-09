"""Manifest CSV -> phoneme labels + cached 42-channel features (sharded, resumable).

Examples
  # LibriSpeech dev set, clean features only
  python scripts/build_features.py --manifest /content/manifests/dev-clean.csv \
      --out /content/drive/MyDrive/phonrec/features --split dev --variants clean --workers 4
  # training set: clean + one augmented copy (gain + noise)
  python scripts/build_features.py --manifest .../train-clean-100.csv --out ... --split train \
      --variants clean aug1 --extra-lexicon /content/librispeech-lexicon.txt --workers 4
  # own recordings (Windows): rows whose 'split' column equals --split
  python scripts/build_features.py --manifest data/manifest.csv --out data/features --split train --filter-split

Every run writes <out>/<split>/phonemize_report.json (OOV words, dropped utterances).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.data.augment import NoiseBank, augment_audio, utt_rng  # noqa: E402
from phonrec.data.manifest import read_manifest  # noqa: E402
from phonrec.data.store import ShardWriter, shard_done  # noqa: E402
from phonrec.frontend.config import FrontendConfig  # noqa: E402
from phonrec.frontend.streaming import extract_features  # noqa: E402
from phonrec.labels import load_label_map  # noqa: E402
from phonrec.phonemize.lexicon import Lexicon  # noqa: E402
from phonrec.phonemize.phonemizer import Phonemizer  # noqa: E402

_W: dict = {}


def load_audio(path: str) -> np.ndarray:
    from phonrec.audio.io import read_wav, resample
    if path.lower().endswith(".flac"):
        import soundfile as sf  # available on Colab; pip install soundfile elsewhere
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
    else:
        x, sr = read_wav(path)
    return resample(x, sr)


def _init(cfg_dict, noise_dir):
    _W["cfg"] = FrontendConfig.from_dict(cfg_dict)
    _W["bank"] = NoiseBank(noise_dir)


def _work(job):
    uid, path, variant = job
    try:
        x = load_audio(path)
        info = {}
        if variant != "clean":
            x, info = augment_audio(x, utt_rng(uid, variant), _W["bank"])
        f = extract_features(x, _W["cfg"])["features"]
        return uid, variant, f, info, None
    except Exception as e:  # noqa: BLE001
        return uid, variant, None, {}, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-dir", default=None, help="base for relative audio paths (default: repo root)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--filter-split", action="store_true", help="only rows whose split column == --split")
    ap.add_argument("--variants", nargs="+", default=["clean"])
    ap.add_argument("--cmudict", default="package", help="'package' (pip cmudict) or a file path")
    ap.add_argument("--overrides", default=str(ROOT / "lexicon/overrides.txt"))
    ap.add_argument("--extra-lexicon", nargs="*", default=[])
    ap.add_argument("--g2p", action="store_true", help="use g2p_en for OOV words if installed")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--wiener", choices=["on", "off"], default="on")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--max-utts", type=int, default=0, help="limit for quick experiments (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    base = Path(args.base_dir) if args.base_dir else ROOT
    rows = read_manifest(args.manifest)
    if args.filter_split:
        rows = [r for r in rows if r.get("split") == args.split]
    rows.sort(key=lambda r: r["recording_id"])
    if args.max_utts:
        # round up to whole shards so a later full run continues with identical shard contents
        n = -(-args.max_utts // args.shard_size) * args.shard_size
        rows = rows[:n]

    lm = load_label_map()
    lex = Lexicon.from_files(args.cmudict, args.overrides, args.extra_lexicon, use_g2p=args.g2p)
    ph = Phonemizer(lex, lm)
    keep, dropped = [], Counter()
    for r in rows:
        res = ph.phonemize(r["transcript"])
        if res.phones is None:
            dropped["oov"] += 1
            r["labels"] = None            # stays in its shard slot, but is skipped
            continue
        r["labels"] = lm.encode(res.phones)
        p = Path(r["audio_path"])
        r["abs_path"] = str(p if p.is_absolute() else base / p)
        keep.append(r)
    out_split = Path(args.out) / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    report = {"manifest": args.manifest, "rows": len(rows), "kept": len(keep), "dropped": dict(dropped),
              "label_sources": dict(lex.source), "oov_top": lex.oov.most_common(200),
              "oov_unique": len(lex.oov)}
    (out_split / "phonemize_report.json").write_text(json.dumps(report, indent=1))
    print(f"[phonemize] {len(keep)}/{len(rows)} utterances kept, {len(lex.oov)} unique OOV words "
          f"(see {out_split / 'phonemize_report.json'})")

    cfg = FrontendConfig(wiener=(args.wiener == "on"))
    (out_split / "frontend.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    max_frames = int(args.max_seconds * 100)
    # shards are defined on the sorted manifest ROWS (not on the kept rows), so their
    # contents do not depend on --max-utts or on lexicon changes elsewhere in the list
    shards = [[r for r in rows[i: i + args.shard_size] if r["labels"] is not None]
              for i in range(0, len(rows), args.shard_size)]
    t0 = time.time()
    errors = []
    pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg.to_dict(), args.noise_dir)) \
        if args.workers > 1 else None
    if pool is None:
        _init(cfg.to_dict(), args.noise_dir)
    try:
        for variant in args.variants:
            vdir = out_split / variant
            for k, shard in enumerate(shards):
                if shard_done(vdir, k):
                    continue
                jobs = [(r["recording_id"], r["abs_path"], variant) for r in shard]
                results = pool.map(_work, jobs, chunksize=16) if pool else map(_work, jobs)
                by_id = {r["recording_id"]: r for r in shard}
                w = ShardWriter(vdir, k)
                for uid, v, f, info, err in results:
                    if err or f is None:
                        errors.append((uid, err))
                        continue
                    r = by_id[uid]
                    n_out = (len(f) + 1) // 2
                    if len(f) > max_frames or n_out < len(r["labels"]):
                        continue   # too long, or too short for CTC
                    meta = {k2: r.get(k2, "") for k2 in ("speaker_id", "session_id", "noise", "distance", "mic")}
                    w.add({"id": uid, "labels": r["labels"], **meta, **info}, f)
                w.close()
                done = sum(len(s) for s in shards[: k + 1])
                print(f"[{variant}] shard {k + 1}/{len(shards)}  ({done} utts, {time.time() - t0:.0f} s)", flush=True)
    finally:
        if pool:
            pool.shutdown()
    if errors:
        (out_split / "errors.json").write_text(json.dumps(errors[:1000], indent=1))
        print(f"[warn] {len(errors)} utterances failed (see errors.json)")
    print("done")


if __name__ == "__main__":
    main()
