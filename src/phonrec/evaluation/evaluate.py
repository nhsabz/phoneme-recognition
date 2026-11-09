"""Evaluate a checkpoint on a feature split: PER, S/D/I, per-phoneme, per-condition,
confusion matrix, latency. Saves log-probs for later calibration (dev split only!).

  python -m phonrec.evaluation.evaluate --ckpt runs/model_a/best.pt --features /content/features \
      --split dev --out reports/model_a_dev
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.store import FeatureStore
from ..decoding.greedy import greedy_decode, strip_silence
from ..evaluation.per import AlignStats, align
from ..training.checkpoint import load_model_bundle


@torch.no_grad()
def run(ckpt, features, split, out, variants=("clean",), device=None, save_logprobs=False, group_by=("noise", "distance")):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, b = load_model_bundle(ckpt, dev)
    labels = b["labels"]
    norm = b["normalizer"]
    store = FeatureStore(features, split, variants)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    total = AlignStats()
    groups: dict[str, AlignStats] = defaultdict(AlignStats)
    per_utt, times, audio_s = [], [], 0.0
    lp_dump = {}
    for uid in store.ids:
        f = norm(store.get(uid))
        x = torch.from_numpy(f).T.unsqueeze(0).to(dev)
        t0 = time.perf_counter()
        lp, _ = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        audio_s += len(f) / 100.0
        lp = lp[0].float().cpu().numpy()
        ref = [labels[i] for i in strip_silence(store.labels(uid))]
        hyp = [labels[i] for i in strip_silence(greedy_decode(lp))]
        st = align(ref, hyp)
        total += st
        meta = store.meta(uid)
        for g in group_by:
            if meta.get(g):
                groups[f"{g}={meta[g]}"] += align(ref, hyp)
        per_utt.append({"id": uid, "per": st.per, "n_ref": st.n_ref, "ref": " ".join(ref), "hyp": " ".join(hyp)})
        if save_logprobs:
            lp_dump[uid] = lp.astype(np.float16)

    phones = labels[2:]
    idx = {p: i for i, p in enumerate(phones)}
    n = len(phones)
    conf = np.zeros((n + 1, n + 1), dtype=np.int64)       # last row/col = gap (insertion/deletion)
    for (r, h), c in total.pairs.items():
        conf[idx.get(r, n), idx.get(h, n)] += c
    per_phone = {}
    for p in phones:
        i = idx[p]
        n_ref = int(conf[i].sum())
        per_phone[p] = {"n_ref": n_ref, "correct": int(conf[i, i]), "sub": int(conf[i, :n].sum() - conf[i, i]),
                        "del": int(conf[i, n]), "ins": int(conf[n, i]),
                        "recall": float(conf[i, i] / n_ref) if n_ref else None}
    top_conf = sorted(((phones[i], phones[j], int(conf[i, j])) for i in range(n) for j in range(n)
                       if i != j and conf[i, j] > 0), key=lambda t: -t[2])[:30]
    res = {"ckpt": str(ckpt), "split": split, "n_utts": len(store.ids), **total.as_dict(),
           "groups": {k: v.as_dict() for k, v in sorted(groups.items())},
           "latency": {"device": str(dev), "mean_ms_per_utt": 1000 * float(np.mean(times)),
                       "rtf": float(sum(times) / max(audio_s, 1e-9)), "audio_hours": audio_s / 3600},
           "top_confusions": top_conf, "per_phone": per_phone, "n_params": b.get("n_params")}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    np.save(out / "confusion.npy", conf)
    (out / "confusion_labels.json").write_text(json.dumps(phones + ["<gap>"]))
    with open(out / "per_utterance.jsonl", "w") as fh:
        for r in sorted(per_utt, key=lambda r: -r["per"]):
            fh.write(json.dumps(r) + "\n")
    if save_logprobs:
        np.savez_compressed(out / "logprobs.npz", **lp_dump)
    print(f"{split}: PER {res['per']:.4f}  (S {res['sub']} D {res['del']} I {res['ins']} / {res['n_ref']})  "
          f"RTF {res['latency']['rtf']:.4f} on {dev}")
    for k, v in res["groups"].items():
        print(f"   {k}: PER {v['per']:.4f} ({v['n_ref']} phonemes)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-logprobs", action="store_true")
    a = ap.parse_args()
    if a.split == "test":
        print("NOTE: test split -> report only. Never tune thresholds, temperature or checkpoints on it.")
    run(a.ckpt, a.features, a.split, a.out, device=a.device, save_logprobs=a.save_logprobs)


if __name__ == "__main__":
    main()
