"""
Hybrid Node Extractor for Sinhala Concept Map Generation.

Combines three complementary signals:
  1. Rule-based POS / chunk patterns    → high recall
  2. Embedding similarity to concept anchors → improved precision / noise filtering
  3. Confidence re-ranker (calibrated weighted sum) → balanced, stable ranking

Usage::

    from hybrid_node_extractor import HybridNodeExtractor
    extractor = HybridNodeExtractor(nlp_engine)
    nodes = extractor.extract(text)   # → List[Dict] sorted by confidence DESC
"""

from __future__ import annotations

import math
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sinhala concept anchor vocabulary
# These serve as the *reference concept space* for embedding scoring.
# A candidate scores high when its embedding is close to any anchor.
# ---------------------------------------------------------------------------
_SINHALA_CONCEPT_ANCHORS: List[str] = [
    # Biology / Science
    "ශාකය", "ජීවියා", "ක්‍රියාවලිය", "ඉන්ද්‍රිය", "පද්ධතිය",
    "ශ්‍රිතය", "ව්‍යුහය", "ලක්ෂණය", "ක්‍රමය", "සංකල්පය",
    "නිෂ්පාදනය", "ශක්තිය", "ද්‍රව්‍යය", "ජලය", "ගසය",
    "ප්‍රතික්‍රියාව", "ශරීරය", "ව්‍යාධිය", "රසායනය",
    # Social Sciences / Economics
    "ආර්ථිකය", "සමාජය", "රටාව", "ප්‍රතිපත්තිය", "නීතිය",
    "අධ්‍යාපනය", "ආයතනය", "ව්‍යාපාරය",
    # Technology / Computing
    "ගණකය", "ජාලය", "මෘදුකාංගය", "දෘඩාංගය", "ඇල්ගොරිතමය",
    "ක්‍රමලේඛනය", "දත්ත", "ආකෘතිය",
    # General academic
    "සිද්ධාන්තය", "නිර්වචනය", "ප්‍රකාශය", "ලක්ෂය", "ගුණාංගය",
    "ගණිතය", "භෞතිකය", "ජීව විද්‍යාව",
]

# Sinhala common noun suffixes (high weight when a word ends with these)
_NOUN_SUFFIX_RE = re.compile(
    r"(ය$|ව$|ම$|ක$|ය්$|ව්$|ද$|ත$|ල$|ලා$|ස$|ස්$|ිය$|ීය$|ාව$|ාය$|ීම$|ීමට$|ේ$|ොව$)"
)

# Verb-form endings that should *penalise* a word
_VERB_ENDING_RE = re.compile(
    r"(නවා$|ෙයි$|ෙනවා$|ිනවා$|ීවා$|ූවා$|ාවා$|ෙනු$|ෙන$|ෙමු$|ෙහෙ$|ෙනවාය$)"
)

# Embedded Latin technical terms (almost always concepts)
_LATIN_TERM_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


class HybridNodeExtractor:
    """
    Hybrid concept node extractor.

    Three-stage pipeline
    --------------------
    1. **Rule layer** – produce candidates from:
         * single-word noun heuristics (suffix patterns, IDF, position)
         * sliding-window multi-word noun chunks
         * embedded Latin technical terms
         * Sinling NP chunker output (optional)

    2. **Embedding layer** – batch-encode all candidates and compute the
       maximum cosine similarity against pre-computed *concept anchor*
       embeddings.  This acts as a semantic plausibility filter that
       suppresses noise words even when they pass rule gates.

    3. **Re-ranker** – combine rule_score, embed_score, frequency and
       positional signal into a single ``confidence`` value:

       .. math::
           c = 0.40 \\cdot r + 0.35 \\cdot e + 0.15 \\cdot f + 0.10 \\cdot p

       Multi-word phrases receive a small multiplicative boost (×1.08).
    """

    # ── Re-ranker weights (must sum to 1.0) ─────────────────────────────────
    _W_RULE: float = 0.40   # rule pattern confidence
    _W_EMBED: float = 0.35  # embedding similarity to concept anchors
    _W_FREQ: float = 0.15   # log-normalised in-text frequency
    _W_POS: float = 0.10    # positional bias (earlier = higher)

    # ── Thresholds ───────────────────────────────────────────────────────────
    _RULE_GATE: float = 0.20       # minimum rule score to enter the pipeline
    _MIN_CONFIDENCE: float = 0.35  # minimum final confidence to keep a node

    def __init__(self, nlp_engine: Any) -> None:
        """
        Parameters
        ----------
        nlp_engine:
            An initialised ``SinhalaNLPEngine`` instance.  The extractor
            borrows its tokenizer, embeddings model, stop-word lists and
            helper methods.
        """
        self.engine = nlp_engine
        self._anchors: List[str] = _SINHALA_CONCEPT_ANCHORS
        self._anchor_embeddings: Optional[np.ndarray] = None
        self._init_anchor_embeddings()

    # =========================================================================
    # Public API
    # =========================================================================

    def extract(self, text: str, max_nodes: int = 40) -> List[Dict[str, Any]]:
        """
        Run the full hybrid pipeline and return ranked concept nodes.

        Each node dict contains:
            ``text``, ``type``, ``importance``, ``confidence``,
            ``rule_score``, ``embed_score``, ``frequency``,
            ``context``, ``offset``, ``normalized``

        Parameters
        ----------
        text:
            Raw Sinhala (or mixed) essay text.
        max_nodes:
            Hard cap on the number of returned nodes.

        Returns
        -------
        List[Dict]
            Sorted by ``confidence`` descending, deduplicated.
        """
        if not text or not text.strip():
            return []

        # 1 ── Rule layer: generate candidates
        candidates = self._rule_extract(text)
        if not candidates:
            return []

        # 2 ── Embedding layer: semantic plausibility scoring
        self._score_with_embeddings(candidates)

        # 3 ── Re-ranker: combine signals
        ranked = self._rerank(candidates)

        # 4 ── Filter & deduplicate
        ranked = [c for c in ranked if c["confidence"] >= self._MIN_CONFIDENCE]
        ranked = self._deduplicate(ranked)

        return ranked[:max_nodes]

    # =========================================================================
    # Stage 1 – Rule extraction
    # =========================================================================

    def _rule_extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Produce concept candidates from rule-based sources.

        Sources (in order, merged into a single candidate pool):
        a) Single-word noun heuristics
        b) Multi-word sliding-window noun chunks (2–4 tokens)
        c) Embedded Latin technical terms
        d) Sinling NP chunker (optional, best-effort)
        """
        engine = self.engine
        sentences: List[Tuple[str, int]] = engine._split_sentences_with_spans(text)

        # Pre-compute frequency maps
        all_tokens = [engine._normalize_term(t) for t in engine._tokenize(text)]
        freq_map: Counter = Counter(t for t in all_tokens if t)
        df_map, total_sents = engine._sentence_document_frequency(sentences)

        candidates: Dict[str, Dict[str, Any]] = {}

        for sentence, sent_start in sentences:
            words = engine._tokenize(sentence)
            n = len(words)

            # ── (a) Single-word candidates ──────────────────────────────────
            for idx, word in enumerate(words):
                if (
                    word in engine.stop_words
                    or len(word) < 3
                    or word in engine.action_verbs
                    or word in engine.weak_words
                    or _VERB_ENDING_RE.search(word)
                ):
                    continue

                rule_score = self._single_word_rule_score(
                    word, sentence, idx, n, freq_map, df_map, total_sents
                )
                if rule_score >= self._RULE_GATE:
                    self._upsert(
                        candidates, word, sentence, sent_start,
                        "concept", rule_score, freq_map, text
                    )

            # ── (b) Multi-word noun chunks ───────────────────────────────────
            for i in range(n):
                if words[i] in engine.stop_words or len(words[i]) < 3:
                    continue
                for j in range(i + 2, min(i + 5, n + 1)):
                    phrase = " ".join(words[i:j])
                    if engine._is_stop_phrase(phrase):
                        continue
                    rule_score = engine._calculate_phrase_importance(phrase, text)
                    # Boost when the tail word carries a noun suffix
                    if _NOUN_SUFFIX_RE.search(words[j - 1]):
                        rule_score = min(1.0, rule_score + 0.08)
                    if rule_score >= self._RULE_GATE:
                        self._upsert(
                            candidates, phrase, sentence, sent_start,
                            "concept_phrase", rule_score, freq_map, text
                        )

            # ── (c) Embedded Latin technical terms ───────────────────────────
            for m in _LATIN_TERM_RE.finditer(sentence):
                term = m.group()
                # Technical terms in a Sinhala essay are almost always concepts
                self._upsert(
                    candidates, term, sentence, sent_start,
                    "technical_term", 0.65, freq_map, text
                )

        # ── (d) Sinling NP chunker (optional) ───────────────────────────────
        self._add_sinling_chunks(text, candidates, freq_map)

        return list(candidates.values())

    # =========================================================================
    # Stage 2 – Embedding similarity
    # =========================================================================

    def _score_with_embeddings(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Batch-encode candidates and compare against concept anchor embeddings.

        Sets ``embed_score`` in [0, 1] for every candidate in-place.
        Falls back to 0.5 (neutral) when embeddings are unavailable.
        """
        if self._anchor_embeddings is None or self.engine.embeddings_model is None:
            for c in candidates:
                c["embed_score"] = 0.5
            return

        try:
            texts = [c["text"] for c in candidates]
            cand_embs: np.ndarray = self.engine.embeddings_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Normalise both matrices for efficient cosine via matmul
            cand_norm = cand_embs / (
                np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8
            )
            anc_norm = self._anchor_embeddings / (
                np.linalg.norm(self._anchor_embeddings, axis=1, keepdims=True) + 1e-8
            )
            # Shape: (n_candidates, n_anchors)
            sims: np.ndarray = cand_norm @ anc_norm.T
            # Best anchor match per candidate  → embed_score in [0, 1]
            max_sims: np.ndarray = sims.max(axis=1)
            for c, sim in zip(candidates, max_sims):
                c["embed_score"] = float(np.clip((float(sim) + 1.0) / 2.0, 0.0, 1.0))

        except Exception as exc:
            logger.warning("Embedding scoring failed: %s", exc)
            for c in candidates:
                c.setdefault("embed_score", 0.5)

    # =========================================================================
    # Stage 3 – Confidence re-ranker
    # =========================================================================

    def _rerank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rule_score, embed_score, freq_score and pos_score into a
        single calibrated ``confidence`` value using the class-level weights.

        Multi-word phrases receive a ×1.08 multiplicative boost because
        multi-word concepts are generally more precise than single tokens.
        Short (≤2 char) leftovers are heavily penalised.
        """
        if not candidates:
            return []

        max_freq = max((c.get("frequency", 1) for c in candidates), default=1) or 1

        for c in candidates:
            # Frequency: log-normalised across the batch
            freq_score = math.log1p(c.get("frequency", 1)) / math.log1p(max_freq)

            # Positional bias: earlier positions score slightly higher
            pos_score = 1.0 - min(c.get("offset", 0), 500) / 500.0

            confidence = (
                self._W_RULE * c.get("rule_score", 0.5)
                + self._W_EMBED * c.get("embed_score", 0.5)
                + self._W_FREQ * freq_score
                + self._W_POS * pos_score
            )

            # Multi-word phrase bonus
            if " " in c["text"]:
                confidence = min(1.0, confidence * 1.08)

            # Short token penalty
            if len(c["text"]) <= 2:
                confidence *= 0.40

            c["confidence"] = round(float(confidence), 4)
            c["importance"] = c["confidence"]  # backward-compat alias

        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _init_anchor_embeddings(self) -> None:
        """Pre-compute anchor embeddings (called once at construction)."""
        if self.engine.embeddings_model is None:
            logger.info(
                "HybridNodeExtractor: embeddings model unavailable; "
                "will use embed_score=0.5 (neutral)."
            )
            return
        try:
            self._anchor_embeddings = self.engine.embeddings_model.encode(
                self._anchors,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            logger.info(
                "HybridNodeExtractor: %d concept anchor embeddings computed.",
                len(self._anchors),
            )
        except Exception as exc:
            logger.warning("Could not compute anchor embeddings: %s", exc)

    def _single_word_rule_score(
        self,
        word: str,
        sentence: str,
        pos_idx: int,
        n_words: int,
        freq_map: Counter,
        df_map: Dict[str, int],
        total_sents: int,
    ) -> float:
        """
        Score a single token using:
        * base importance heuristic from the engine
        * frequency bonus
        * IDF boost
        * noun-suffix bonus
        """
        engine = self.engine
        score = engine._calculate_word_importance(word, sentence, pos_idx, n_words)
        if score == 0.0:
            return 0.0

        norm = engine._normalize_term(word)

        # Frequency bonus
        count = freq_map.get(norm, 1)
        if count > 1:
            score += min(0.20, 0.06 * (count - 1))

        # IDF boost: distinctive terms score higher
        df = df_map.get(norm, 0)
        if total_sents > 0 and df > 0:
            idf = math.log((1 + total_sents) / (1 + df)) + 1.0
            score = min(1.0, score * min(1.35, 0.85 + idf * 0.12))

        # Noun-suffix pattern bonus
        if _NOUN_SUFFIX_RE.search(word):
            score = min(1.0, score + 0.07)

        return round(min(1.0, max(0.0, score)), 4)

    def _upsert(
        self,
        candidates: Dict[str, Dict[str, Any]],
        text: str,
        sentence: str,
        sent_start: int,
        entity_type: str,
        rule_score: float,
        freq_map: Counter,
        full_text: str,
    ) -> None:
        """Insert a candidate into the pool, or update its rule_score if higher."""
        key = self.engine._normalize_term(text) or text
        if key in candidates:
            if rule_score > candidates[key]["rule_score"]:
                candidates[key]["rule_score"] = rule_score
            return

        norm = self.engine._normalize_term(text)
        offset = self.engine._find_offset(sentence, text, sent_start, [])
        candidates[key] = {
            "text": text,
            "type": entity_type,
            "rule_score": rule_score,
            "embed_score": 0.5,   # filled in Stage 2
            "confidence": 0.0,    # filled in Stage 3
            "importance": 0.0,    # alias set in Stage 3
            "frequency": full_text.count(text) or 1,
            "context": sentence,
            "offset": offset,
            "normalized": norm,
        }

    def _add_sinling_chunks(
        self,
        text: str,
        candidates: Dict[str, Dict[str, Any]],
        freq_map: Counter,
    ) -> None:
        """Add noun phrase candidates from the Sinling chunker (best-effort)."""
        try:
            from sinling import SinhalaPhraseChunker  # type: ignore

            chunker = SinhalaPhraseChunker()
            engine = self.engine
            for sentence, sent_start in engine._split_sentences_with_spans(text):
                for chunk in chunker.chunk(sentence):
                    if chunk.get("type") != "NP":
                        continue
                    phrase = chunk.get("text", "").strip()
                    if len(phrase) < 3 or engine._is_stop_phrase(phrase):
                        continue
                    rule_score = min(
                        1.0,
                        engine._calculate_phrase_importance(phrase, text) + 0.12,
                    )
                    if rule_score >= self._RULE_GATE:
                        self._upsert(
                            candidates, phrase, sentence, sent_start,
                            "np_chunk", rule_score, freq_map, text
                        )
        except Exception:
            pass  # Sinling is optional

    def _deduplicate(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove candidates that are:
        * strict sub-strings of a higher-ranked candidate, OR
        * exact duplicates by normalised key.
        """
        seen_keys: set = set()
        seen_texts: List[str] = []
        result: List[Dict[str, Any]] = []

        for c in ranked:
            key = c.get("normalized") or self.engine._normalize_term(c["text"]) or c["text"]
            if key in seen_keys:
                continue
            # Suppress if a longer, already-accepted phrase subsumes this text
            if any(c["text"] in st and st != c["text"] for st in seen_texts):
                continue
            seen_keys.add(key)
            seen_texts.append(c["text"])
            result.append(c)

        return result
