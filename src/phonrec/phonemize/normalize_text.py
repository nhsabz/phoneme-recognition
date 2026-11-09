"""Deterministic transcript normalisation.

Rules (documented so labels are reproducible):
* Unicode quotes/dashes are mapped to ASCII; text is upper-cased.
* The token ``<pause>`` (case-insensitive) marks a long pause and is kept as a
  special token; it becomes SIL in the phoneme sequence.
* Integers 0-9999 are spelled out ("42" -> "FORTY TWO"). Other digit strings
  are kept as-is and will be reported as OOV.
* Hyphens, slashes and underscores split words. All other punctuation is
  removed. Apostrophes inside words are kept (DON'T, JOHN'S); leading and
  trailing apostrophes are stripped.
"""
from __future__ import annotations

import re
import unicodedata

PAUSE_TOKEN = "<PAUSE>"

_ONES = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN".split()
_TENS = "_ _ TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()


def number_to_words(n: int) -> str:
    if not 0 <= n <= 9999:
        raise ValueError(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " HUNDRED" + ("" if r == 0 else " " + number_to_words(r))
    th, r = divmod(n, 1000)
    return _ONES[th] + " THOUSAND" + ("" if r == 0 else " " + number_to_words(r))


_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'",
    "“": '"', "”": '"', "–": "-", "—": "-",
})


def normalize_transcript(text: str) -> list[str]:
    """Return a list of word tokens (and PAUSE_TOKEN)."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = re.sub(r"<\s*pause\s*>", " \x00 ", text, flags=re.IGNORECASE)
    text = text.upper()
    text = re.sub(r"[-/_]", " ", text)
    out: list[str] = []
    for raw in text.split():
        if raw == "\x00":
            out.append(PAUSE_TOKEN)
            continue
        tok = re.sub(r"[^A-Z0-9']", "", raw).strip("'")
        if not tok:
            continue
        if tok.isdigit():
            n = int(tok)
            if n <= 9999:
                out.extend(number_to_words(n).split())
                continue
        out.append(tok)
    return out
