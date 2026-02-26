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
    6. Conservative spelling correction (high-confidence Sinhala forms)
    7. Token boundary recovery (merged/split token repair)
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

_SINHALA_TOKEN_RE = re.compile(r"[\u0D80-\u0DFF]+")

_LATIN_DIGIT_TO_SINHALA_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z0-9])(?=[\u0D80-\u0DFF])|(?<=[\u0D80-\u0DFF])(?=[A-Za-z0-9])")

# High-confidence lexical corrections (common OCR / keyboard variants)
_SPELLING_CORRECTIONS = {
    "සමග": "සමඟ",
    "සමගින්": "සමඟින්",
    "පිලිබඳ": "පිළිබඳ",
    "පිලිබඳව": "පිළිබඳව",
    "පිලිබද": "පිළිබඳ",
    "පිලිබදව": "පිළිබඳව",
    "කෙසේවෙතත්": "කෙසේ වෙතත්",
    "එබැවින්ම": "එබැවින්",
    "ලැබෙනවා": "ලැබේ",
}

# Function words frequently glued to neighboring tokens in OCR/user text
_BOUNDARY_HINTS = sorted([
    "සඳහා", "පිළිබඳව", "පිළිබඳ", "මගින්", "සමඟ", "නිසා", "හේතුවෙන්",
    "ලෙස", "යනු", "තුළ", "අතර", "වෙත", "විසින්",
], key=len, reverse=True)


def _correct_spelling(text: str) -> str:
    """Apply conservative token-level spelling fixes."""
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        tok = match.group(0)
        return _SPELLING_CORRECTIONS.get(tok, tok)

    return _SINHALA_TOKEN_RE.sub(_replace, text)


def _recover_token_boundaries(text: str) -> str:
    """
    Recover common token-boundary issues that fragment Sinhala concepts.

    Steps:
      1) Insert boundaries at Sinhala↔Latin/digit transitions.
      2) Split tokens that accidentally glue common Sinhala function words.
      3) Join runs of single-Sinhala-char tokens (OCR split artifacts).
    """
    if not text:
        return text

    fixed = _LATIN_DIGIT_TO_SINHALA_BOUNDARY_RE.sub(" ", text)

    # Insert a boundary before closed-class hints when merged (e.g., "අධ්‍යාපනයසඳහා").
    for hint in _BOUNDARY_HINTS:
        if len(hint) < 3:
            continue
        pattern = re.compile(rf"(?<=[\u0D80-\u0DFF])({re.escape(hint)})(?=$|[\s\u0D80-\u0DFF])")
        fixed = pattern.sub(r" \1", fixed)

    # Rebuild token stream, joining long runs of single Sinhala letters.
    tokens = fixed.split()
    rebuilt = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if len(tok) == 1 and _SINHALA_TOKEN_RE.fullmatch(tok):
            run = [tok]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and _SINHALA_TOKEN_RE.fullmatch(tokens[j]):
                run.append(tokens[j])
                j += 1
            if len(run) >= 3:
                rebuilt.append("".join(run))
            else:
                rebuilt.extend(run)
            i = j
            continue
        rebuilt.append(tok)
        i += 1

    return " ".join(rebuilt)


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

    # 8. Spelling correction (high-confidence only)
    text = _correct_spelling(text)

    # 9. Token boundary recovery
    text = _recover_token_boundaries(text)

    # Final whitespace cleanup after token-boundary edits
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()

    return text
