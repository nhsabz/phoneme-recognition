"""CTC training with resume, AMP, gradient clipping, bucketing and SpecAugment.

  python -m phonrec.training.train --config configs/train_a.json \
      --features /content/features --out /content/drive/MyDrive/phonrec/runs/model_a

Re-running the same command after a Colab disconnect resumes from <out>/last.pt.
Outputs in <out>: last.pt, best.pt (lowest dev PER), history.csv, config.json, stats.json.

Nondeterminism: seeds are fixed, but CUDA's CTC backward and some cuDNN kernels
are not bit-deterministic, so two GPU runs can differ slightly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.store import FeatureStore
from ..decoding.greedy import greedy_decode, strip_silence
from ..evaluation.per import corpus_per
from ..frontend.normalize import FeatureNormalizer
from ..labels import load_label_map
from ..models.blocks import count_params
from ..models.models import build_model
from .checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from .dataset import BucketBatchSampler, PhonemeDataset, make_collate

DEFAULTS = {
    "model": "a", "model_kwargs": {}, "seed": 1234,
    "train_split": "train", "train_variants": ["clean", "aug1"],
    "dev_split": "dev", "dev_variants": ["clean"],
    "batch_size": 48, "epochs": 40, "lr": 1e-3, "weight_decay": 0.01, "warmup_epochs": 1.0,
    "grad_clip": 5.0, "amp": True, "num_workers": 2, "patience": 8,
    "spec_augment": {"n_freq": 2, "max_f": 8, "n_time": 2, "max_t": 40},
    "max_train_utts": 0, "log_every": 100, "init_from": None,
}


def seed_everything(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_normalizer(store: FeatureStore, max_utts: int = 4000, seed: int = 0) -> FeatureNormalizer:
    """Mean/std from (a random subset of) the TRAINING split, clean variant only."""
    rng = random.Random(seed)
    ids = store.ids if len(store.ids) <= max_utts else rng.sample(store.ids, max_utts)
    n = FeatureNormalizer()
    for u in ids:
        n.accumulate(store.get(u, "clean" if "clean" in store.variants else None))
    return n.finalize()


@torch.no_grad()
def evaluate_ctc(model, loader, device, ctc, amp: bool, silence_id=1):
    model.eval()
    tot_loss, n_utt, refs, hyps, refs_sil, hyps_sil = 0.0, 0, [], [], [], []
    for x, lens, targets, tl, _ in loader:
        x, lens = x.to(device), lens.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            lp, ol = model(x, lens)
        lp = lp.float()
        loss = ctc(lp.transpose(0, 1), targets.to(device), ol, tl.to(device))
        tot_loss += float(loss) * len(tl)
        n_utt += len(tl)
        lp_np, ol_np = lp.cpu().numpy(), ol.cpu().numpy()
        off = 0
        tgt = targets.tolist()
        for i, n_i in enumerate(tl.tolist()):
            ref = tgt[off: off + n_i]
            off += n_i
            hyp = greedy_decode(lp_np[i, : ol_np[i]])
            refs_sil.append(ref)
            hyps_sil.append(hyp)
            refs.append(strip_silence(ref, silence_id))
            hyps.append(strip_silence(hyp, silence_id))
    st, st_sil = corpus_per(refs, hyps), corpus_per(refs_sil, hyps_sil)
    model.train()
    return {"loss": tot_loss / max(n_utt, 1), "per": st.per, "per_with_sil": st_sil.per,
            "sub": st.sub, "del": st.dele, "ins": st.ins, "n_ref": st.n_ref}


def lr_lambda_factory(warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    return f


def train(cfg: dict, features: str, out: str, device: str | None = None, resume: bool = True):
    cfg = {**DEFAULTS, **cfg}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(cfg["amp"]) and dev.type == "cuda"
    seed_everything(cfg["seed"])
    lm = load_label_map()

    train_store = FeatureStore(features, cfg["train_split"], cfg["train_variants"])
    dev_store = FeatureStore(features, cfg["dev_split"], cfg["dev_variants"])
    if cfg["max_train_utts"]:
        train_store.ids = train_store.ids[: cfg["max_train_utts"]]
    fe_cfg = json.loads((Path(features) / cfg["train_split"] / "frontend.json").read_text())

    last = out / "last.pt"
    ck = load_checkpoint(last) if (resume and last.exists()) else None
    if ck is not None:
        norm = FeatureNormalizer(ck["norm_mean"], ck["norm_std"])
    elif cfg.get("init_from"):
        init = load_checkpoint(cfg["init_from"])
        norm = FeatureNormalizer(init["norm_mean"], init["norm_std"])   # keep pretraining stats
    else:
        norm = compute_normalizer(train_store, seed=cfg["seed"])

    ds_tr = PhonemeDataset(train_store, norm, train=True, seed=cfg["seed"])
    ds_dv = PhonemeDataset(dev_store, norm, train=False)
    samp_tr = BucketBatchSampler(ds_tr.lengths(), cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    samp_dv = BucketBatchSampler(ds_dv.lengths(), cfg["batch_size"], shuffle=False)
    dl_tr = DataLoader(ds_tr, batch_sampler=samp_tr, num_workers=cfg["num_workers"],
                       collate_fn=make_collate(True, cfg["spec_augment"]), pin_memory=dev.type == "cuda",
                       persistent_workers=cfg["num_workers"] > 0)
    dl_dv = DataLoader(ds_dv, batch_sampler=samp_dv, num_workers=cfg["num_workers"],
                       collate_fn=make_collate(False), pin_memory=dev.type == "cuda")

    model = build_model(cfg["model"], **cfg["model_kwargs"]).to(dev)
    if ck is None and cfg.get("init_from"):
        model.load_state_dict(load_checkpoint(cfg["init_from"])["model"])
        print(f"initialised weights from {cfg['init_from']}")
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)   # no weight decay on BN / biases
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    steps_per_epoch = len(samp_tr)
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(max(1, int(cfg["warmup_epochs"] * steps_per_epoch)), total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc = torch.nn.CTCLoss(blank=lm.blank_id, reduction="mean", zero_infinity=True)

    start_epoch, best_per, bad_epochs, step = 0, float("inf"), 0, 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, best_per, bad_epochs, step = ck["epoch"] + 1, ck["best_per"], ck["bad_epochs"], ck["step"]
        print(f"resumed from {last} at epoch {start_epoch} (best dev PER {best_per:.4f})")

    n_params = count_params(model)
    (out / "config.json").write_text(json.dumps({**cfg, "n_params": n_params, "device": str(dev),
                                                 "train_utts": len(ds_tr), "dev_utts": len(ds_dv),
                                                 "frontend": fe_cfg}, indent=1))
    print(f"model {cfg['model']}: {n_params:,} params | train {len(ds_tr)} utts | dev {len(ds_dv)} utts "
          f"| {steps_per_epoch} steps/epoch | device {dev} | amp {use_amp}")

    hist_path = out / "history.csv"
    fields = ["epoch", "step", "train_loss", "dev_loss", "dev_per", "dev_per_with_sil", "sub", "del", "ins",
              "lr", "epoch_time_s", "best"]
    if not hist_path.exists():
        with open(hist_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def bundle(epoch):
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch, "step": step,
                "best_per": best_per, "bad_epochs": bad_epochs, "config": cfg, "model_name": cfg["model"],
                "model_kwargs": cfg["model_kwargs"], "labels": list(lm.labels), "frontend": fe_cfg,
                "norm_mean": norm.mean.tolist(), "norm_std": norm.std.tolist(), "n_params": n_params}

    model.train()
    for epoch in range(start_epoch, cfg["epochs"]):
        samp_tr.set_epoch(epoch)
        t0, run_loss, n_b = time.time(), 0.0, 0
        for x, lens, targets, tl, _ in dl_tr:
            x, lens, targets, tl = x.to(dev, non_blocking=True), lens.to(dev), targets.to(dev), tl.to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                lp, ol = model(x, lens)
            loss = ctc(lp.float().transpose(0, 1), targets, ol, tl)   # CTC in float32
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += float(loss)
            n_b += 1
            if step % cfg["log_every"] == 0:
                print(f"  ep {epoch} step {step} loss {run_loss / n_b:.3f} grad {float(gn):.2f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        ev = evaluate_ctc(model, dl_dv, dev, ctc, use_amp, lm.silence_id)
        is_best = ev["per"] < best_per
        if is_best:
            best_per, bad_epochs = ev["per"], 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "step": step, "train_loss": round(run_loss / max(n_b, 1), 4),
               "dev_loss": round(ev["loss"], 4), "dev_per": round(ev["per"], 4),
               "dev_per_with_sil": round(ev["per_with_sil"], 4), "sub": ev["sub"], "del": ev["del"],
               "ins": ev["ins"], "lr": f"{sched.get_last_lr()[0]:.2e}",
               "epoch_time_s": round(time.time() - t0, 1), "best": int(is_best)}
        with open(hist_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {epoch}: train {row['train_loss']} | dev loss {row['dev_loss']} | dev PER "
              f"{row['dev_per']:.4f} (S {ev['sub']} D {ev['del']} I {ev['ins']}) | {row['epoch_time_s']} s"
              + ("  *best*" if is_best else ""), flush=True)
        b = bundle(epoch)
        if is_best:
            save_checkpoint(out / "best.pt", b)
        save_checkpoint(last, b)
        if bad_epochs >= cfg["patience"]:
            print(f"early stop: no dev PER improvement for {bad_epochs} epochs")
            break
    print(f"finished. best dev PER {best_per:.4f} -> {out / 'best.pt'}")
    return best_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=json_value")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    train(cfg, a.features, a.out, a.device, resume=not a.no_resume)


if __name__ == "__main__":
    main()
