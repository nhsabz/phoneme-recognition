"""PC recording utility: 16 kHz, mono, 16-bit WAV + transcript + manifest row.

Usage (Windows, inside the project venv):
    python scripts/record.py --list-devices
    python scripts/record.py --speaker spk01 --session s001 --noise quiet \
        --distance near --mic "laptop-internal" --prompts data/prompts/prompts_v1.txt

Per prompt: press Enter to START, Enter to STOP. Then [Enter]=keep, r=redo, s=skip, q=quit.
The mic is opened at its native rate (often 48 kHz on Windows) and resampled
to 16 kHz with the same code the live app uses (phonrec.audio.io.resample).

Outputs
    data/recordings/<session>/<recording_id>.wav   (16 kHz mono int16)
    data/recordings/<session>/<recording_id>.txt   (transcript)
    data/recordings/<session>/session.json         (session metadata)
    data/manifest.csv                              (one row per recording; split filled later)
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.audio.io import TARGET_SR, level_report, resample, to_mono, write_wav  # noqa: E402

MANIFEST_FIELDS = ["recording_id", "audio_path", "transcript", "speaker_id", "session_id",
                   "noise", "distance", "mic", "device", "native_sr", "duration_s", "split", "created"]


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print("\nDefault input device:", sd.default.device[0])


def record_until_enter(device, native_sr: int) -> np.ndarray:
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def cb(indata, frames, t, status):
        if status:
            print(f"  [audio status] {status}", file=sys.stderr)
        q.put(indata.copy())

    with sd.InputStream(device=device, samplerate=native_sr, channels=1, dtype="float32", callback=cb):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        print("  ● recording... press Enter to stop")
        t0 = time.time()
        while not stop.is_set():
            time.sleep(0.05)
            if time.time() - t0 > 60:
                print("  (60 s limit reached)")
                break
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return to_mono(np.concatenate(chunks)) if chunks else np.zeros(0, np.float32)


def append_manifest(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", default=None, help="input device index or name (default: system default)")
    ap.add_argument("--speaker", default="spk01")
    ap.add_argument("--session", required="--list-devices" not in sys.argv)
    ap.add_argument("--noise", choices=["quiet", "noisy"], default="quiet")
    ap.add_argument("--distance", choices=["near", "far"], default="near",
                    help="near ~30 cm, far ~1.5 m (keep the rule fixed within a session)")
    ap.add_argument("--mic", default="laptop-internal")
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts/prompts_v1.txt"))
    ap.add_argument("--out", default=str(ROOT / "data/recordings"))
    ap.add_argument("--manifest", default=str(ROOT / "data/manifest.csv"))
    ap.add_argument("--start", type=int, default=0, help="prompt index to start from")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001
        sys.exit(f"sounddevice not available: {e}\nInstall with: pip install sounddevice")
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    try:
        info = sd.query_devices(device, "input")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"No usable input device ({e}). Run with --list-devices and pass --device N.")
    native_sr = int(info["default_samplerate"])

    prompts = [ln.strip() for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")]
    sess_dir = Path(args.out) / args.session
    sess_dir.mkdir(parents=True, exist_ok=True)
    meta = {"session_id": args.session, "speaker_id": args.speaker, "noise": args.noise,
            "distance": args.distance, "mic": args.mic, "device": info["name"], "native_sr": native_sr,
            "target_sr": TARGET_SR, "started": datetime.now().isoformat(timespec="seconds"),
            "prompts_file": str(args.prompts)}
    (sess_dir / "session.json").write_text(json.dumps(meta, indent=2))
    print(f"Device: {info['name']} @ {native_sr} Hz -> saved as {TARGET_SR} Hz mono 16-bit")
    print(f"Session {args.session} | speaker {args.speaker} | {args.noise} | {args.distance}\n")

    i = args.start
    while i < len(prompts):
        text = prompts[i]
        rid = f"{args.session}_{args.speaker}_{i:04d}"
        print(f"[{i + 1}/{len(prompts)}]  \"{text}\"")
        cmd = input("  Enter = start, s = skip, q = quit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        x = record_until_enter(device, native_sr)
        y = resample(x, native_sr, TARGET_SR)
        lv = level_report(y)
        print(f"  {len(y) / TARGET_SR:.2f} s | peak {lv['peak_dbfs']:.1f} dBFS | rms {lv['rms_dbfs']:.1f} dBFS")
        for w in lv["warnings"]:
            print("  !!", w)
        if len(y) < 0.3 * TARGET_SR:
            print("  too short, redo")
            continue
        ans = input("  Enter = keep, r = redo, s = skip > ").strip().lower()
        if ans == "r":
            continue
        if ans == "s":
            i += 1
            continue
        wav = sess_dir / f"{rid}.wav"
        write_wav(wav, y, TARGET_SR)
        (sess_dir / f"{rid}.txt").write_text(text + "\n", encoding="utf-8")
        append_manifest(Path(args.manifest), {
            "recording_id": rid, "audio_path": wav.relative_to(ROOT).as_posix(), "transcript": text,
            "speaker_id": args.speaker, "session_id": args.session, "noise": args.noise,
            "distance": args.distance, "mic": args.mic, "device": info["name"], "native_sr": native_sr,
            "duration_s": f"{len(y) / TARGET_SR:.3f}", "split": "",
            "created": datetime.now().isoformat(timespec="seconds")})
        i += 1
    print("Done. Manifest:", args.manifest)


if __name__ == "__main__":
    main()
