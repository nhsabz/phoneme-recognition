"""Zip the code (not data, venv or checkpoints) for upload to Google Drive.

  python scripts/make_code_zip.py          -> phonrec_code.zip in the repo root
Upload it to  My Drive/phonrec/phonrec_code.zip  (the notebook unzips it).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", "checkpoints", "smoke_out", "smoke_out_pytest",
             "recordings", "features", "librispeech", "noise", ".git", "logs"}
SKIP_EXT = {".wav", ".flac", ".npy", ".npz", ".pt", ".zip", ".gz"}

out = ROOT / "phonrec_code.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_file() and p.suffix not in SKIP_EXT:
            z.write(p, Path("phoneme-recognition") / rel)
            n += 1
print(f"wrote {out} ({n} files, {out.stat().st_size / 1024:.0f} KB)")
