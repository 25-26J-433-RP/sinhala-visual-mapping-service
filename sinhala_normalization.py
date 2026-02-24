"""
Sinhala Text Normalization Layer.

Provides a single entry-point ``normalize_sinhala_text`` that applies a
sequence of Unicode and orthographic normalisations optimised for Sinhala
concept-map extraction:

  1. NFC Unicode normalisation (composite character forms)
  2. Zero-width character removal
  3. Punctuation normalisation (Sinhala danda → period, curly quotes, etc.)
  4. Whitespace collapsing
  5. Repeated character reduction (≥3 repetitions → 2)
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled patterns (built once at import time for performance)
# ---------------------------------------------------------------------------

# Zero-width chars: ZWSP, ZWNJ, ZWJ, BOM, word-joiner, soft-hyphen
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF\u2060\u00AD]")

# Sinhala danda (U+0DF4) → Latin full-stop for consistent sentence splitting
_DANDA_RE = re.compile(r"\u0DF4")

# Curly / typographic quotes → straight quotes
_CURLY_QUOTE_RE = re.compile(r"[\u2018\u2019\u201A\u201B]")
_CURLY_DQUOTE_RE = re.compile(r"[\u201C\u201D\u201E\u201F\u2039\u203A]")

# Repeated identical characters (≥3 → 2); handles noise like 'ශශශ'
_REPEAT_RE = re.compile(r"(.)\1{2,}")

# Multiple whitespace (space, tab, non-breaking space) → single space
_WHITESPACE_RE = re.compile(r"[\t\u00A0\u202F\u2009\u3000 ]{2,}")

# Standalone punctuation runs that add no semantic content
_PUNCT_NOISE_RE = re.compile(r"[।|]{2,}")


def normalize_sinhala_text(text: str) -> str:
    """
    Apply all Sinhala-specific normalisation steps to *text*.

    Parameters
    ----------
    text:
        Raw input string (may be Sinhala, mixed Sinhala-English, or empty).

    Returns
    -------
    str
        Normalised text ready for tokenisation and concept extraction.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # 1. NFC Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Remove zero-width characters
    text = _ZERO_WIDTH_RE.sub("", text)

    # 3. Sinhala danda → full-stop
    text = _DANDA_RE.sub(".", text)

    # 4. Normalise quotes
    text = _CURLY_QUOTE_RE.sub("'", text)
    text = _CURLY_DQUOTE_RE.sub('"', text)

    # 5. Reduce repeated characters
    text = _REPEAT_RE.sub(r"\1\1", text)

    # 6. Collapse redundant punctuation runs
    text = _PUNCT_NOISE_RE.sub(".", text)

    # 7. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()

    return text
