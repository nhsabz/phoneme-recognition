"""P6: fit the confidence temperature on DEV log-probs and build the 0-7 confidence table.

  python scripts/calibrate.py --logprobs reports/model_a_v1_dev/logprobs.npz \
      --per-utt reports/model_a_v1_dev/per_utterance.jsonl --out reports/model_a_v1_calib

Inputs come from `phonrec.evaluation.evaluate --split dev --save-logprobs`. Refuses test-split inputs.
Honest held-out check: T is fit on one half of the dev speakers and ECE is reported on the other
half; the final T (written to calibration.json, used by the live app) is then fit on all of dev.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.evaluation.calibration import (binary_nll, collect_tokens, ece, fit_temperature,  # noqa: E402
                                            level_table, speaker_half)
from phonrec.evaluation.per import align  # noqa: E402
from phonrec.labels import load_label_map  # noqa: E402


def summary(ts, T):
    p = ts.confidences(T)
    return {"T": T, "n_tokens": int(len(p)), "accuracy": float(ts.correct.mean()),
            "mean_conf": float(p.mean()), "nll": binary_nll(p, ts.correct), "ece": ece(p, ts.correct)}


def plot(tables: dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed -> no reliability.png (numbers are in calibration.json)")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    for name, tab in tables.items():
        pts = [(r["mean_conf"], r["accuracy"]) for r in tab if r["n"]]
        ax.plot(*zip(*pts), "o-", label=name)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy of emitted phonemes")
    ax.set_title("Reliability (held-out dev speakers)"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprobs", required=True)
    ap.add_argument("--per-utt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "test" in Path(a.logprobs).parent.name or "test" in Path(a.per_utt).parent.name:
        sys.exit("refusing: calibration must be fit on DEV outputs, never on test")

    lm = load_label_map()
    idx = {p: i for i, p in enumerate(lm.labels)}
    refs, hyps_saved = {}, {}
    with open(a.per_utt) as f:
        for line in f:
            r = json.loads(line)
            refs[r["id"]] = [idx[p] for p in r["ref"].split()]
            hyps_saved[r["id"]] = r["hyp"]
    z = np.load(a.logprobs)
    missing = [u for u in refs if u not in z.files]
    if missing:
        sys.exit(f"{len(missing)} utterances in per_utterance.jsonl have no log-probs (e.g. {missing[:3]})")
    lps = {u: z[u].astype(np.float32) for u in refs}

    ts = collect_tokens(lps, refs, lm.blank_id, lm.silence_id)
    # consistency check: our greedy hypotheses must equal the ones evaluate.py scored. The dump is
    # float16, so frames whose top-2 log-probs round to a tie can flip; allow that tiny drift only.
    hyp_now = {}
    for u, k in zip(ts.uid, ts.cls):
        hyp_now.setdefault(u, []).append(lm.labels[k])
    bad = [u for u in refs if " ".join(hyp_now.get(u, [])) != hyps_saved[u]]
    edits = sum(align(hyps_saved[u].split(), hyp_now.get(u, [])).errors for u in bad)
    n_tok = sum(len(h.split()) for h in hyps_saved.values())
    if edits > 0.001 * n_tok:
        sys.exit(f"{len(bad)} hypotheses differ from evaluate.py output ({edits} token edits; e.g. {bad[:3]}); "
                 "wrong files?")
    if bad:
        print(f"[info] float16 rounding: {len(bad)} utts differ from evaluate.py by {edits} of {n_tok} tokens")

    half = np.array([speaker_half(u) for u in ts.uid])
    fit_part, held = ts.subset(half == 0), ts.subset(half == 1)
    T_half = fit_temperature(fit_part)
    T_all = fit_temperature(ts)
    p_held_1, p_held_T = held.confidences(1.0), held.confidences(T_half)
    p_all = ts.confidences(T_all)
    res = {
        "source": {"logprobs": a.logprobs, "per_utt": a.per_utt, "n_utts": len(refs),
                   "speakers_fit": int(len({u.split('-')[0] for u in fit_part.uid})),
                   "speakers_held": int(len({u.split('-')[0] for u in held.uid}))},
        "definition": "confidence = max posterior of the emitted phoneme over its greedy CTC span; "
                      "correct = aligned match vs reference; SIL excluded; level = floor(8p) clipped to 7",
        "temperature": T_all,
        "held_out_check": {"T_fit_on_half": T_half, "before_T1": summary(held, 1.0),
                           "after": summary(held, T_half),
                           "levels_before": level_table(p_held_1, held.correct),
                           "levels_after": level_table(p_held_T, held.correct)},
        "all_dev": {"before_T1": summary(ts, 1.0), "after": summary(ts, T_all),
                    "levels": level_table(p_all, ts.correct)},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=1))
    hc = res["held_out_check"]
    print(f"tokens {len(ts.cls)} from {len(refs)} dev utts | emitted-phoneme accuracy {ts.correct.mean():.4f}")
    print(f"held-out speakers: T fit on other half = {T_half:.3f} | ECE {hc['before_T1']['ece']:.4f} -> "
          f"{hc['after']['ece']:.4f} | NLL {hc['before_T1']['nll']:.4f} -> {hc['after']['nll']:.4f}")
    print(f"final T (all dev) = {T_all:.3f}  -> {out / 'calibration.json'}")
    print("level  claims        n      mean_conf  accuracy   (held-out, after T)")
    for r in hc["levels_after"]:
        mc = "-" if r["mean_conf"] is None else f"{r['mean_conf']:.3f}"
        ac = "-" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        print(f"  {r['level']}    {r['claims'][0]:.3f}-{r['claims'][1]:.3f}  {r['n']:7d}  {mc:>9}  {ac:>8}")
    plot({"T=1": hc["levels_before"], f"T={T_half:.2f}": hc["levels_after"]}, out / "reliability.png")


if __name__ == "__main__":
    main()
