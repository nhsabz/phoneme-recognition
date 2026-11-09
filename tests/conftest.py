import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

FIXTURE_DICT = """\
;;; tiny CMUdict-format fixture for tests (real CMUdict is downloaded separately)
HELLO  HH AH0 L OW1
HELLO(2)  HH EH0 L OW1
WORLD  W ER1 L D
THE  DH AH0
CAT  K AE1 T
SAT  S AE1 T
DON'T  D OW1 N T
FORTY  F AO1 R T IY0
TWO  T UW1
TOP  T AA1 P
PAT  P AE1 T
"""


@pytest.fixture
def cmu_path(tmp_path):
    p = tmp_path / "cmudict-fixture.txt"
    p.write_text(FIXTURE_DICT, encoding="latin-1")
    return p
