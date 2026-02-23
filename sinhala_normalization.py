"""
Sinhala-specific normalization utilities for text preprocessing.
Unifies ය/යා variants, punctuation, zero-width, Latin-Sinhala mix, and common spelling variants.
"""
import re
import unicodedata

# Common spelling variants (expand as needed)
SPELLING_VARIANTS = [
    (r"ය(ා)?", "යා"),  # unify ය/යා to යා
    # Add more rules as needed
]

# Zero-width characters in Unicode
ZERO_WIDTH_CHARS = [
    '\u200B', '\u200C', '\u200D', '\uFEFF'
]

# Sinhala and Latin punctuation normalization
PUNCTUATION_MAP = {
    '“': '"', '”': '"', '‘': "'", '’': "'",
    '–': '-', '—': '-', '…': '...',
    '।': '.',  # Sinhala danda to period
}

LATIN_TO_SINHALA_MAP = {
    # Add mappings if needed for common Latin chars to Sinhala
}

def normalize_sinhala_text(text: str) -> str:
    """Apply Sinhala-specific normalization to text."""
    if not text:
        return text
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    # Remove zero-width chars
    for zw in ZERO_WIDTH_CHARS:
        text = text.replace(zw, '')
    # Normalize punctuation
    for k, v in PUNCTUATION_MAP.items():
        text = text.replace(k, v)
    # Latin-Sinhala mix: (optional, can add more logic)
    for k, v in LATIN_TO_SINHALA_MAP.items():
        text = text.replace(k, v)
    # Unify spelling variants
    for pattern, repl in SPELLING_VARIANTS:
        text = re.sub(pattern, repl, text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
