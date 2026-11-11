"""P7 live phoneme recogniser (Windows PC mic, or a WAV file through the identical live path).

  python scripts/live.py --ckpt checkpoints/best.pt --calib reports/model_a_v1_calib/calibration.json
  python scripts/live.py ... --prompt "The tall dog dug a deep pit by the old tree."   # say it, Ctrl+C
  python scripts/live.py ... --wav data/recordings/s000/s000_spk01_0000.wav           # replay a file
  python scripts/live.py --list-devices

Prints each phoneme as PHONEME(level), level 0-7 = calibrated confidence, about 0.3 s after it was
spoken (the model waits for 260 ms of following audio; it never guesses future audio). A new line
starts after each pause (soft VAD). Stop with Ctrl+C (or --seconds N).

Log per run in logs/live/<time>/: audio16k.wav (exactly what was recognised; replay with --wav),
tokens.jsonl, summary.json (latency, processing speed, overflows, and with --prompt the reference,
PER and the individual errors).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.audio.io import TARGET_SR, read_wav, write_wav  # noqa: E402
from phonrec.evaluation.per import align  # noqa: E402
from phonrec.streaming.live import LivePipeline  # noqa: E402
from phonrec.training.checkpoint import load_model_bundle  # noqa: E402


def reference_phones(prompt: str) -> list[str] | None:
    from phonrec.labels import load_label_map
    from phonrec.phonemize.lexicon import Lexicon
    from phonrec.phonemize.phonemizer import Phonemizer
    lm = load_label_map()
    res = Phonemizer(Lexicon.from_files("package", ROOT / "lexicon/overrides.txt"), lm).phonemize(prompt)
    if res.phones is None:
        print(f"[warn] prompt has words not in the lexicon: {res.oov_words} -> no PER")
        return None
    return [p for p in res.phones if p != lm.silence]


class Clock:
    """Maps a native-rate sample index to the wall-clock time its audio block reached the app."""

    def __init__(self):
        self.ends, self.times = [], []

    def add(self, end_sample: int, t: float):
        self.ends.append(end_sample)
        self.times.append(t)

    def time_of(self, sample: int) -> float | None:
        i = int(np.searchsorted(self.ends, sample))
        return self.times[i] if i < len(self.times) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/best.pt"))
    ap.add_argument("--calib", default=None, help="calibration.json from scripts/calibrate.py (dev-fitted T)")
    ap.add_argument("--device", default=None, help="input device index or name (default: system default)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--wav", default=None, help="run a WAV file through the live path instead of the mic")
    ap.add_argument("--prompt", default=None, help="sentence you will say: logs PER and errors at the end")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--block-ms", type=int, default=50)
    ap.add_argument("--show-sil", action="store_true")
    ap.add_argument("--log-dir", default=str(ROOT / "logs/live"))
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        print("\nDefault input device:", sd.default.device[0])
        return

    model, bundle = load_model_bundle(a.ckpt, "cpu")
    if bundle["model_name"] not in ("a", "model_a"):
        sys.exit("live streaming supports Model A only (Model B's BiGRU needs future audio)")
    T, calib_src = 1.0, None
    if a.calib:
        T, calib_src = float(json.loads(Path(a.calib).read_text())["temperature"]), a.calib
    else:
        print("[warn] no --calib: temperature 1.0 (uncalibrated)")
    ref = reference_phones(a.prompt) if a.prompt else None

    out = Path(a.log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    if a.wav:
        x_file, sr = read_wav(a.wav)
        source = {"wav": a.wav}
    else:
        import sounddevice as sd
        device = int(a.device) if a.device and a.device.isdigit() else a.device
        info = sd.query_devices(device, "input")
        sr = int(info["default_samplerate"])
        source = {"device": info["name"], "native_sr": sr}
    pipe = LivePipeline(model, bundle, sr, temperature=T, blank_id=0)
    sil = bundle["labels"][1]
    block = max(1, sr * a.block_ms // 1000)

    clock, tokens, audio16, lat, proc_s = Clock(), [], [], [], 0.0
    n_native, overflows = 0, 0
    print(f"model {bundle['model_name'].upper()} ({bundle.get('n_params', 0):,} params) | T={T:.3f} | "
          f"{source} | lookahead {pipe.net.look * 10} ms")
    print("listening... (Ctrl+C to stop)\n" if not a.wav else "")

    def handle(res, now):
        audio16.append(res["audio16"])
        for tk in res["tokens"]:
            rec = {"phoneme": tk.label, "level": tk.level, "conf": round(tk.conf, 4),
                   "start_s": round(tk.start_s, 3), "end_s": round(tk.end_s, 3)}
            t_cap = clock.time_of(int(tk.end_s * sr))
            if t_cap is not None and not a.wav:
                rec["latency_s"] = round(now - t_cap, 3)    # end of phoneme audio -> shown
                lat.append(now - t_cap)
            tokens.append(rec)
            if tk.label != sil or a.show_sil:
                print(f"{tk.label}({tk.level})", end=" ", flush=True)
        if any(kind == "end" for kind, _ in res["vad_events"]):
            print(flush=True)

    t_start = time.monotonic()
    try:
        if a.wav:
            for i in range(0, len(x_file), block):
                c = x_file[i: i + block]
                n_native += len(c)
                clock.add(n_native, time.monotonic())
                t0 = time.perf_counter()
                res = pipe.push(c)
                proc_s += time.perf_counter() - t0
                handle(res, time.monotonic())
        else:
            q: queue.Queue = queue.Queue()

            def cb(indata, frames, t, status):
                q.put((indata[:, 0].copy(), time.monotonic(), bool(status.input_overflow)))

            with sd.InputStream(device=device, samplerate=sr, channels=1, dtype="float32",
                                blocksize=block, callback=cb):
                while not (a.seconds and time.monotonic() - t_start > a.seconds):
                    try:
                        c, t_cb, ovf = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    overflows += ovf
                    n_native += len(c)
                    clock.add(n_native, t_cb)
                    t0 = time.perf_counter()
                    res = pipe.push(c)
                    proc_s += time.perf_counter() - t0
                    handle(res, time.monotonic())
    except KeyboardInterrupt:
        pass
    t0 = time.perf_counter()
    res = pipe.flush()
    proc_s += time.perf_counter() - t0
    handle(res, time.monotonic())
    print()

    dur = n_native / sr
    a16 = np.concatenate(audio16) if audio16 else np.zeros(0, np.float32)
    write_wav(out / "audio16k.wav", a16, TARGET_SR)
    with open(out / "tokens.jsonl", "w") as f:
        for r in tokens:
            f.write(json.dumps(r) + "\n")
    hyp = [r["phoneme"] for r in tokens if r["phoneme"] != sil]
    levels = np.bincount([r["level"] for r in tokens if r["phoneme"] != sil], minlength=8).tolist()
    summary = {"ckpt": a.ckpt, "calibration": calib_src, "temperature": T, "source": source,
               "duration_s": round(dur, 3), "n_phonemes": len(hyp), "level_histogram_0_7": levels,
               "processing_rtf": round(proc_s / max(dur, 1e-9), 4), "input_overflows": overflows,
               "algorithmic_lookahead_ms": pipe.net.look * 10, "hyp": " ".join(hyp)}
    if lat:
        summary["latency_s"] = {"mean": round(float(np.mean(lat)), 3), "median": round(float(np.median(lat)), 3),
                                "p95": round(float(np.percentile(lat, 95)), 3), "max": round(float(np.max(lat)), 3)}
    if ref is not None:
        st = align(ref, hyp)
        summary.update({"prompt": a.prompt, "ref": " ".join(ref), **{f"ref_{k}": v for k, v in st.as_dict().items()},
                        "errors": sorted(([r or "-", h or "-", c] for (r, h), c in st.pairs.items() if r != h),
                                         key=lambda e: -e[2])})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{dur:.1f} s audio | {len(hyp)} phonemes | processing RTF {summary['processing_rtf']} | "
          f"overflows {overflows}" + (f" | latency median {summary['latency_s']['median']} s, "
                                      f"p95 {summary['latency_s']['p95']} s" if lat else ""))
    if ref is not None:
        print(f"REF {summary['ref']}\nHYP {summary['hyp']}\nPER {summary['ref_per']:.3f} "
              f"(S {summary['ref_sub']} D {summary['ref_del']} I {summary['ref_ins']} / {summary['ref_n_ref']})")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
