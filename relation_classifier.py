"""
Relation Classifier for Sinhala Concept Map Generation.

Implements a two-stage pipeline for relationship extraction:

  Stage 1 – Candidate generation
  --------------------------------
  Shortlist node pairs using a multi-criterion gate:
    • Same-sentence co-occurrence
    • Adjacent-sentence co-occurrence
    • High embedding similarity (≥ threshold)
    • Shared character n-gram root

  Stage 2 – Relation classification
  ------------------------------------
  For each shortlisted pair, extract a multi-dimensional feature vector
  and pass it through a calibrated soft-score classifier that outputs one
  of four canonical relation types with a confidence score:

    • is-a           (definitional / taxonomic)
    • part-of        (compositional / membership)
    • cause-effect   (causal / functional)
    • related-to     (semantic proximity / conjunction / co-occurrence)

  The classifier is feature-weighted rather than black-box, making its
  decisions interpretable and easy to tune via the weight tables below.
  The design is also annotation-ready: weight vectors can be replaced by
  trained logistic-regression coefficients once labeled data is available.

Usage::

    from relation_classifier import RelationClassifier
    rc = RelationClassifier(nlp_engine)
    rels = rc.classify(text, entities)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical relation types
# ---------------------------------------------------------------------------
REL_IS_A = "is-a"
REL_PART_OF = "part-of"
REL_CAUSE = "cause-effect"
REL_RELATED = "related-to"
CANONICAL_TYPES = [REL_IS_A, REL_PART_OF, REL_CAUSE, REL_RELATED]

# ---------------------------------------------------------------------------
# Per-type prototype phrases (Sinhala)
# Used to build "type-space" embedding axes for the between-span feature.
# ---------------------------------------------------------------------------
_TYPE_PROTOTYPES: Dict[str, List[str]] = {
    REL_IS_A: [
        "යනු", "හෙවත්", "කියන", "ලෙස හඳුන්වේ", "ලෙස නිර්වචනය",
        "ලෙස අර්ථ දැක්වේ", "ගණ්‍ය", "ශ්‍රේණිය",
    ],
    REL_PART_OF: [
        "කොටසකි", "අංශය", "අයත්", "ඇතුළත්", "කොටසක් වේ",
        "ශාඛාව", "කොටසක්", "දෙකොටස", "අභ්‍යන්තර",
    ],
    REL_CAUSE: [
        "නිසා", "හේතුවෙන්", "ප්‍රතිඵලයෙන්", "ජනනය කරයි",
        "ඇති කරයි", "සිදු කරයි", "ප්‍රේරණය", "ඉදිරියට",
    ],
    REL_RELATED: [
        "සම්බන්ධ", "ආශ්‍රිත", "සහ", "සමඟ", "ආශ්‍රිතය",
        "සමාන", "සමකාලීන",
    ],
}

# ---------------------------------------------------------------------------
# Lexical marker lists (mirrors nlp_engine but scoped here for independence)
# ---------------------------------------------------------------------------
_MARKERS: Dict[str, List[str]] = {
    REL_IS_A: [
        "යනු", "හෙවත්", "කියන", "අර්ථය", "අර්ථයෙන්",
        "හැඳින්වෙයි", "ලෙස", "ලෙසින්",
        "උදාහරණ", "උදාහරණයක්", "උදා:",
    ],
    REL_PART_OF: [
        "අංශය", "කොටස", "අයත්", "අංග", "අභ්‍යන්තර",
        "භාගය", "ශාඛාව", "කොටසකි",
    ],
    REL_CAUSE: [
        "නිසා", "හේතුවෙන්", "ප්‍රතිඵලයෙන්",
        "ප්‍රතිඵලයක් ලෙස", "ඒ හේතුවෙන්",
        "කරන", "ඇති කරයි", "සිදු කරයි",
    ],
    REL_RELATED: [
        "සහ", "හා", "මෙන්ම", "සමඟ", "ද", "සම්බන්ධ",
        "ආශ්‍රිත", "ආශ්‍රිතය",
    ],
}

# ---------------------------------------------------------------------------
# Calibrated weight matrix
# Rows → canonical types  |  Columns → feature dimensions (see _feature_names)
#
# Feature dimensions (14 total):
#  [0]  lex_is_a_between      marker presence between entities
#  [1]  lex_part_between
#  [2]  lex_cause_between
#  [3]  lex_related_between
#  [4]  lex_is_a_context       marker presence in full sentence context
#  [5]  lex_part_context
#  [6]  lex_cause_context
#  [7]  lex_related_context
#  [8]  embed_sim              cosine(entity1, entity2)  in [0,1]
#  [9]  between_is_a_sim       cosine(between_span, is-a prototype)
#  [10] between_part_sim
#  [11] between_cause_sim
#  [12] same_sentence          1.0 / 0.0
#  [13] distance_norm          1 - dist/max_dist, in [0,1]
# ---------------------------------------------------------------------------
_WEIGHT_MATRIX = np.array([
    # is-a   part   cause  rel    is-a*  part*  cause* rel*   esim   b_is  b_pt  b_ca  ssent  dist
    [  1.40,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00,  0.00,  0.00,  1.00, 0.00, 0.00, 0.30,  0.10],  # is-a
    [  0.00,  1.40,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00,  0.00,  0.00, 1.00, 0.00, 0.30,  0.10],  # part-of
    [  0.00,  0.00,  1.40,  0.00,  0.00,  0.00,  0.90,  0.00,  0.00,  0.00, 0.00, 1.00, 0.30,  0.10],  # cause-effect
    [  0.00,  0.00,  0.00,  0.70,  0.00,  0.00,  0.00,  0.40,  0.50,  0.00, 0.00, 0.00, 0.40,  0.30],  # related-to
], dtype=np.float64)  # shape (4, 14)

# Bias terms per class (log-prior)
_BIAS = np.array([-0.40, -0.50, -0.40, 0.10], dtype=np.float64)

# ---------------------------------------------------------------------------
# Candidate generation parameters
# ---------------------------------------------------------------------------
_EMBED_SIM_GATE: float = 0.40      # minimum cosine to shortlist a cross-sentence pair
_NGRAM_SIM_GATE: float = 0.45      # character n-gram similarity gate
_MAX_SENT_DISTANCE: int = 2        # max sentence index gap for cross-sentence pairs
_MAX_OFFSET_DISTANCE: int = 600    # max character offset gap for proximity pairs
_MAX_PAIRS_PER_NODE: int = 6       # cap pairs per source node to avoid O(n²) explosion


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _char_ngram_sim(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    ga = {a[i: i + n] for i in range(max(1, len(a) - n + 1))}
    gb = {b[i: i + n] for i in range(max(1, len(b) - n + 1))}
    u = len(ga | gb)
    return len(ga & gb) / u if u else 0.0


class RelationClassifier:
    """
    Two-stage relation classifier:
      1. Candidate pair generation (multi-criterion shortlisting)
      2. Feature-based multi-class relation classification with softmax output

    Parameters
    ----------
    nlp_engine :
        Initialised ``SinhalaNLPEngine`` instance.
    """

    def __init__(self, nlp_engine: Any) -> None:
        self.engine = nlp_engine
        self._proto_embs: Optional[Dict[str, np.ndarray]] = None
        self._init_prototype_embeddings()

    # =========================================================================
    # Public API
    # =========================================================================

    def classify(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run the full two-stage pipeline and return relationship dicts.

        Each relationship dict contains:
          ``source``, ``target``, ``type``, ``confidence``,
          ``context``, ``feature_scores`` (per-type softmax probabilities)

        Parameters
        ----------
        text :
            Pre-normalised source text.
        entities :
            Concept nodes returned by the hybrid extractor.

        Returns
        -------
        List[Dict]
            Deduplicated relationships sorted by confidence descending.
        """
        if not entities or len(entities) < 2:
            return []

        sentences = self.engine._split_sentences_with_spans(text)

        # ── Stage 1: candidate pairs ──────────────────────────────────────────
        candidates = self._generate_candidates(entities, sentences, text)

        # ── Stage 2: classify each candidate ─────────────────────────────────
        relationships: Dict[tuple, Dict[str, Any]] = {}
        for pair in candidates:
            rel = self._classify_pair(pair, text)
            if rel is None:
                continue
            threshold = self.engine.rel_thresholds.get(rel["type"], 0.45)
            if rel["confidence"] < threshold:
                continue
            key = tuple(sorted([rel["source"], rel["target"]]) + [rel["type"]])
            if key not in relationships or relationships[key]["confidence"] < rel["confidence"]:
                relationships[key] = rel

        result = sorted(relationships.values(), key=lambda r: r["confidence"], reverse=True)
        return result

    # =========================================================================
    # Stage 1 – Candidate generation
    # =========================================================================

    def _generate_candidates(
        self,
        entities: List[Dict[str, Any]],
        sentences: List[Tuple[str, int]],
        full_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Shortlist entity pairs using four complementary criteria.

        Returns a list of candidate dicts, each containing:
          e1, e2, context, same_sentence, sent_distance, offset_distance
        """
        # Build sentence-index mapping: entity → sentence index
        sent_ranges = [(start, start + len(s), idx, s)
                       for idx, (s, start) in enumerate(sentences)]

        def sent_idx_of(entity: Dict) -> Tuple[int, str]:
            off = entity.get("offset", -1)
            for start, end, idx, s in sent_ranges:
                if start <= off < end:
                    return idx, s
            return -1, entity.get("context", "")

        seen: set = set()
        candidates: List[Dict[str, Any]] = []
        pair_count: Dict[str, int] = {}  # source key → count

        # Pre-compute embeddings for all entities (single batch call)
        entity_embs = self._batch_embed([e["text"] for e in entities])

        n = len(entities)
        for i in range(n):
            src_key = entities[i].get("normalized") or entities[i]["text"]

            for j in range(i + 1, n):
                if pair_count.get(src_key, 0) >= _MAX_PAIRS_PER_NODE:
                    break

                e1, e2 = entities[i], entities[j]
                if e1["text"] == e2["text"]:
                    continue

                pair_key = tuple(sorted([e1["text"], e2["text"]]))
                if pair_key in seen:
                    continue

                si, s1_text = sent_idx_of(e1)
                sj, s2_text = sent_idx_of(e2)

                same_sent = (si == sj and si >= 0)
                sent_dist = abs(si - sj) if si >= 0 and sj >= 0 else 99
                off_dist = abs(e1.get("offset", 0) - e2.get("offset", 0))

                # Gate 1: same sentence
                qualifies = same_sent
                # Gate 2: adjacent sentence
                if not qualifies and sent_dist <= _MAX_SENT_DISTANCE:
                    qualifies = True
                # Gate 3: embedding similarity
                if not qualifies and entity_embs is not None:
                    esim = self._cosine(entity_embs[i], entity_embs[j])
                    if esim >= _EMBED_SIM_GATE:
                        qualifies = True
                # Gate 4: character n-gram root similarity
                if not qualifies:
                    ng = _char_ngram_sim(e1["text"], e2["text"])
                    if ng >= _NGRAM_SIM_GATE:
                        qualifies = True

                if not qualifies:
                    continue
                if off_dist > _MAX_OFFSET_DISTANCE and not same_sent:
                    continue

                # Choose best context sentence
                context = s1_text if same_sent else (s1_text or s2_text)

                emb_sim = (
                    self._cosine(entity_embs[i], entity_embs[j])
                    if entity_embs is not None else 0.50
                )

                seen.add(pair_key)
                pair_count[src_key] = pair_count.get(src_key, 0) + 1
                candidates.append({
                    "e1": e1, "e2": e2,
                    "context": context,
                    "same_sentence": same_sent,
                    "sent_distance": sent_dist,
                    "offset_distance": off_dist,
                    "embed_sim": float(emb_sim),
                    "e1_emb_idx": i,
                    "e2_emb_idx": j,
                    "_entity_embs": entity_embs,
                })

        return candidates

    # =========================================================================
    # Stage 2 – Feature extraction + classification
    # =========================================================================

    def _classify_pair(
        self, pair: Dict[str, Any], full_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for one candidate pair and run the soft classifier.

        Returns a relationship dict or None if the pair should be discarded.
        """
        e1: Dict = pair["e1"]
        e2: Dict = pair["e2"]
        context: str = pair["context"]
        entity_embs = pair.get("_entity_embs")
        i, j = pair["e1_emb_idx"], pair["e2_emb_idx"]

        # ── Extract text between the two entities in the context sentence ────
        between = self._between_text(e1["text"], e2["text"], context)

        # ── Lexical marker features ───────────────────────────────────────────
        lex_between = self._marker_scores(between)
        lex_context = self._marker_scores(context)

        # ── Embedding features ────────────────────────────────────────────────
        embed_sim = float(pair.get("embed_sim", 0.50))

        between_type_sims = self._between_type_similarities(between)

        # ── Structural features ───────────────────────────────────────────────
        same_sent = 1.0 if pair["same_sentence"] else 0.0
        dist_norm = max(0.0, 1.0 - pair["offset_distance"] / _MAX_OFFSET_DISTANCE)

        # ── Assemble 14-dim feature vector ────────────────────────────────────
        feat = np.array([
            lex_between[REL_IS_A],      # [0]
            lex_between[REL_PART_OF],   # [1]
            lex_between[REL_CAUSE],     # [2]
            lex_between[REL_RELATED],   # [3]
            lex_context[REL_IS_A],      # [4]
            lex_context[REL_PART_OF],   # [5]
            lex_context[REL_CAUSE],     # [6]
            lex_context[REL_RELATED],   # [7]
            embed_sim,                  # [8]
            between_type_sims[REL_IS_A],    # [9]
            between_type_sims[REL_PART_OF], # [10]
            between_type_sims[REL_CAUSE],   # [11]
            same_sent,                  # [12]
            dist_norm,                  # [13]
        ], dtype=np.float64)

        # ── Softmax score per type ────────────────────────────────────────────
        logits = _WEIGHT_MATRIX @ feat + _BIAS        # shape (4,)
        probs = _softmax(logits)                       # shape (4,)

        best_idx = int(np.argmax(probs))
        rel_type = CANONICAL_TYPES[best_idx]
        confidence = float(probs[best_idx])

        # Reject if classifier is very uncertain (all probs ≈ uniform) or best
        # type probability is too low even before the per-type threshold gate.
        if confidence < 0.30:
            return None

        return {
            "source": e1["text"],
            "target": e2["text"],
            "type": rel_type,
            "confidence": round(confidence, 4),
            "context": context,
            "feature_scores": {
                REL_IS_A:    round(float(probs[0]), 4),
                REL_PART_OF: round(float(probs[1]), 4),
                REL_CAUSE:   round(float(probs[2]), 4),
                REL_RELATED: round(float(probs[3]), 4),
            },
        }

    # =========================================================================
    # Feature helpers
    # =========================================================================

    def _marker_scores(self, text: str) -> Dict[str, float]:
        """Return presence score (0–1) of each type's markers in *text*."""
        scores: Dict[str, float] = {}
        for rtype, markers in _MARKERS.items():
            hits = sum(1 for m in markers if m in text)
            # Normalise: full presence if ≥2 markers hit; saturate
            scores[rtype] = min(1.0, hits / 2.0)
        return scores

    def _between_type_similarities(self, between: str) -> Dict[str, float]:
        """
        Embed the between-span text and compute cosine similarity
        against each type's prototype embedding centroid.

        Falls back to 0.5 (neutral) when embeddings are unavailable or
        the between-span is empty.
        """
        default = {t: 0.50 for t in CANONICAL_TYPES}
        if not between.strip() or self._proto_embs is None:
            return default
        try:
            span_emb = self.engine.embeddings_model.encode(
                [between], show_progress_bar=False, convert_to_numpy=True
            )[0]
            span_norm = span_emb / (np.linalg.norm(span_emb) + 1e-8)
            result = {}
            for rtype in CANONICAL_TYPES:
                proto = self._proto_embs.get(rtype)
                if proto is None:
                    result[rtype] = 0.50
                else:
                    sim = float(np.dot(span_norm, proto))  # proto already normalised
                    result[rtype] = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
            return result
        except Exception as exc:
            logger.debug("between-type similarity failed: %s", exc)
            return default

    def _between_text(self, text1: str, text2: str, sentence: str) -> str:
        """
        Extract the substring of *sentence* that lies between text1 and text2.

        Handles the case where one entity is a substring of the other:
        if text1 is contained in text2 (or vice-versa), return the
        non-overlapping remainder so causal markers embedded in compound
        phrase-nodes are still captured.
        """
        # Substring / prefix case: return the non-shared portion
        if text1 in text2:
            return text2.replace(text1, '', 1).strip()
        if text2 in text1:
            return text1.replace(text2, '', 1).strip()

        idx1 = sentence.find(text1)
        idx2 = sentence.find(text2)
        if idx1 == -1 or idx2 == -1:
            return ""
        lo = min(idx1, idx2) + len(text1 if idx1 < idx2 else text2)
        hi = max(idx1, idx2)
        return sentence[lo:hi].strip() if lo < hi else ""

    # =========================================================================
    # Embedding helpers
    # =========================================================================

    def _init_prototype_embeddings(self) -> None:
        """Pre-compute normalised centroid embeddings for each relation type."""
        if self.engine.embeddings_model is None:
            logger.info("RelationClassifier: no embeddings model; "
                        "between-type similarities will be 0.5.")
            return
        try:
            result: Dict[str, np.ndarray] = {}
            for rtype, phrases in _TYPE_PROTOTYPES.items():
                embs = self.engine.embeddings_model.encode(
                    phrases, show_progress_bar=False, convert_to_numpy=True
                )
                centroid = embs.mean(axis=0)
                result[rtype] = centroid / (np.linalg.norm(centroid) + 1e-8)
            self._proto_embs = result
            logger.info("RelationClassifier: prototype embeddings precomputed "
                        "for %d relation types.", len(result))
        except Exception as exc:
            logger.warning("Could not compute prototype embeddings: %s", exc)

    def _batch_embed(
        self, texts: List[str]
    ) -> Optional[np.ndarray]:
        """Batch-encode texts; return normalised embedding matrix or None."""
        if self.engine.embeddings_model is None or not texts:
            return None
        try:
            embs: np.ndarray = self.engine.embeddings_model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            return embs / norms
        except Exception as exc:
            logger.warning("Batch embedding failed: %s", exc)
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))
