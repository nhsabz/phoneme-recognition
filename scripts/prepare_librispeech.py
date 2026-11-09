"""Write unified manifest CSVs for extracted LibriSpeech subsets.

  python scripts/prepare_librispeech.py --root /content/data --subsets train-clean-100 dev-clean test-clean \
      --out /content/manifests
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phonrec.data.librispeech import librispeech_rows  # noqa: E402
from phonrec.data.manifest import write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--subsets", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
for s in a.subsets:
    rows = librispeech_rows(a.root, s)
    write_manifest(Path(a.out) / f"{s}.csv", rows)
    spk = len({r["speaker_id"] for r in rows})
    print(f"{s}: {len(rows)} utterances, {spk} speakers -> {Path(a.out) / f'{s}.csv'}")
