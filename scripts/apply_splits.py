"""Fill the `split` column of data/manifest.csv from configs/splits.json (whole sessions, decision D13).

  python scripts/apply_splits.py              # writes data/manifest.csv (old copy: manifest.csv.bak)
  python scripts/apply_splits.py --check      # only report, write nothing

Also removes duplicate rows of a recording_id (the recorder appends a new row when a take is redone
in a resumed session; the WAV on disk is the last take, so the last row is kept), and warns when a
session is missing from splits.json, a WAV is missing, or a session read the wrong prompt list.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.data.manifest import read_manifest, write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifest.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/splits.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    sessions = json.loads(Path(a.splits).read_text(encoding="utf-8"))["sessions"]
    rows = read_manifest(a.manifest)
    last = {}
    for r in rows:
        last[r["recording_id"]] = r            # later rows win
    n_dup = len(rows) - len(last)
    prompts = {}
    for name in {s["prompts"] for s in sessions.values()}:
        p = ROOT / "data/prompts" / name
        prompts[name] = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.startswith("#")}

    out, problems = [], []
    for rid, r in last.items():
        s = sessions.get(r["session_id"])
        if s is None:
            problems.append(f"{rid}: session {r['session_id']} not in splits.json -> split left empty")
            r["split"] = ""
        else:
            r["split"] = s["split"]
            if r["transcript"] not in prompts[s["prompts"]]:
                problems.append(f"{rid}: transcript is not from {s['prompts']} (wrong --prompts for this session?)")
            for k in ("noise", "distance", "speaker_id"):
                want = s.get("speaker" if k == "speaker_id" else k)
                if want and r.get(k) != want:
                    problems.append(f"{rid}: {k}={r.get(k)!r} but splits.json says {want!r}")
        if not (ROOT / r["audio_path"]).exists():
            problems.append(f"{rid}: missing audio {r['audio_path']}")
        out.append(r)

    cnt = Counter((r["split"] or "-", r["session_id"]) for r in out)
    for (sp, sess), n in sorted(cnt.items()):
        print(f"  {sp:8s} {sess}: {n} recordings")
    print(f"{len(out)} recordings ({n_dup} duplicate rows removed)")
    for p in problems:
        print("  !!", p)
    if a.check:
        return
    shutil.copy(a.manifest, a.manifest + ".bak")
    write_manifest(a.manifest, out)
    print(f"wrote {a.manifest} (backup {a.manifest}.bak)")


if __name__ == "__main__":
    main()
