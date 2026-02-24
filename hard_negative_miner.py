"""
Hard Negative Miner for Sinhala Concept-Extractor Training.

Problem
-------
A concept-extraction model trained only on positive examples easily learns to
associate *any* noun-like, semantically-rich Sinhala word with the ``concept``
label.  Words that are topically close to real concepts (e.g. "ක්‍රමයකි" – "in
a way", "ජනතාවට" – "to the people") fire false positives at inference time
because they live in the same embedding neighbourhood as labelled concepts.

Solution
--------
Mine "hard negatives" from the annotated training corpus:
  * Every unlabelled (O-tagged) Sinhala word/phrase whose embedding has a high
    cosine similarity (≥ ``sim_threshold``) to the concept-anchor vocabulary is
    a *hard negative* – it looks like a concept but is not.

Three augmentation strategies are then applied to inject these into the training
set so the model explicitly learns the boundary:

  1. **Upsample** – duplicate the original sentence that contains the hard
     negative.  The model sees the difficult context twice.

  2. **Inject** – replace a real concept span in *another* training sentence
     with the hard-negative phrase, re-tagging it as O.  This forces the model
     to predict "not a concept" for an embedding it might naively classify as
     positive.

  3. **Concatenate** – join a concept-rich sentence with the hard-negative
     sentence (separated by a full stop) to build denser contrast contexts.
     Only other concept labels shift into the new combined text; the hard
     negative itself stays O.

Usage (inside train_concept_extractor.py)::

    from hard_negative_miner import HardNegativeMiner

    miner = HardNegativeMiner(sim_threshold=0.55, max_hard_negatives=120)
    extra_samples = miner.mine_from_jsonl("concept_train.jsonl")
    # extra_samples is a list of {"text": ..., "tags": [...]} dicts
    # mix with the original training samples before creating the Dataset
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concept anchor vocabulary — identical to hybrid_node_extractor._SINHALA_CONCEPT_ANCHORS
# so the two components agree on what "looks like a concept".
# ---------------------------------------------------------------------------
_CONCEPT_ANCHORS: List[str] = [
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

# Sinhala Unicode block for word extraction
_SINHALA_WORD_RE = re.compile(r"[\u0D80-\u0DFF][\u0D80-\u0DFF\u200C\u200D]*")

# Minimum length for an O-span to be considered as a hard-negative candidate
_MIN_SPAN_LEN: int = 3


class HardNegativeMiner:
    """
    Mine semantically confusable non-concept spans from annotated Sinhala data
    and produce augmented training samples that sharpen the concept boundary.

    Parameters
    ----------
    sim_threshold : float
        Cosine similarity (to the nearest concept anchor) required for an
        O-tagged span to be classified as a hard negative.  Range [0, 1].
        Lower → more negatives (noisier); higher → fewer but purer.
        Default: 0.55
    max_hard_negatives : int
        Hard cap on the total number of hard-negative candidates to process.
        Caps O(n²) augmentation cost.  Default: 150
    upsample_factor : int
        How many duplicate copies to produce per hard-negative via strategy 1.
        Default: 1
    inject_per_hard_neg : int
        How many injection augmentations (strategy 2) to generate per hard
        negative.  Default: 2
    concat_per_hard_neg : int
        How many concatenation augmentations (strategy 3) to generate per hard
        negative.  Default: 1
    seed : int
        Random seed for reproducible augmentation selection.
    """

    _EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        sim_threshold: float = 0.55,
        max_hard_negatives: int = 150,
        upsample_factor: int = 1,
        inject_per_hard_neg: int = 2,
        concat_per_hard_neg: int = 1,
        seed: int = 42,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.max_hard_negatives = max_hard_negatives
        self.upsample_factor = upsample_factor
        self.inject_per_hard_neg = inject_per_hard_neg
        self.concat_per_hard_neg = concat_per_hard_neg
        random.seed(seed)
        np.random.seed(seed)

        self._embed_model = None
        self._anchor_embs: Optional[np.ndarray] = None
        self._load_embeddings()

    # =========================================================================
    # Public API
    # =========================================================================

    def mine_from_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load annotated JSONL data, mine hard negatives and return augmented
        training samples ready to be mixed with the original training set.

        Each returned sample is a dict with keys:
            ``text``  – raw text string
            ``tags``  – list of per-character label strings
                        (``"O"``, ``"B-concept"``, ``"B-multiword-concept"``)

        Parameters
        ----------
        jsonl_path : str
            Path to the annotated JSONL file used for training.

        Returns
        -------
        List[Dict]
            New synthetic samples (does NOT include the original samples; add
            them yourself before constructing the Dataset).
        """
        path = Path(jsonl_path)
        if not path.exists():
            logger.warning("Hard negative miner: file not found – %s", jsonl_path)
            return []

        raw: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

        if not raw:
            return []

        logger.info(
            "HardNegativeMiner: loaded %d samples from %s", len(raw), jsonl_path
        )

        return self._generate_augmented(raw)

    # =========================================================================
    # Core pipeline
    # =========================================================================

    def _generate_augmented(
        self, raw_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Full mining + augmentation pipeline."""

        # ── Step 1: Extract labelled concept spans and O-spans ────────────────
        concept_spans: List[Dict[str, Any]] = []   # confirmed concept regions
        o_span_candidates: List[Dict[str, Any]] = []  # unlabelled spans

        for idx, sample in enumerate(raw_samples):
            text = sample["text"]
            labels: List[Tuple[int, int, str]] = [
                (s, e, l) for s, e, l in sample.get("labels", [])
            ]

            # Which character offsets are covered by a concept annotation?
            labeled_chars: set = set()
            for s, e, _ in labels:
                labeled_chars.update(range(s, e))

            # Collect concept spans (for injection augmentation)
            for s, e, lbl in labels:
                concept_spans.append(
                    {"text": text[s:e], "sample_idx": idx, "start": s, "end": e, "label": lbl}
                )

            # Collect O-span candidates: Sinhala words not covered by any label
            for m in _SINHALA_WORD_RE.finditer(text):
                ws, we = m.start(), m.end()
                word = m.group()
                if (
                    len(word) < _MIN_SPAN_LEN
                    or labeled_chars.intersection(range(ws, we))
                ):
                    continue
                o_span_candidates.append(
                    {"text": word, "sample_idx": idx, "start": ws, "end": we}
                )

        logger.info(
            "HardNegativeMiner: %d concept spans, %d O-span candidates",
            len(concept_spans),
            len(o_span_candidates),
        )

        # ── Step 2: Identify hard negatives via embedding similarity ──────────
        hard_negatives = self._find_hard_negatives(o_span_candidates)
        logger.info(
            "HardNegativeMiner: found %d hard negatives (threshold=%.2f)",
            len(hard_negatives),
            self.sim_threshold,
        )

        if not hard_negatives:
            return []

        # ── Step 3: Augmentation ──────────────────────────────────────────────
        augmented: List[Dict[str, Any]] = []

        for hn in hard_negatives:
            orig_sample = raw_samples[hn["sample_idx"]]

            # Strategy 1: Upsample the containing sentence
            for _ in range(self.upsample_factor):
                augmented.append(self._build_tagged_sample(orig_sample))

            # Strategy 2: Inject hard negative into other samples' concept slots
            donor_pool = [
                cp for cp in concept_spans if cp["sample_idx"] != hn["sample_idx"]
            ]
            if donor_pool:
                donors = random.sample(
                    donor_pool, min(self.inject_per_hard_neg, len(donor_pool))
                )
                for donor_span in donors:
                    donor_raw = raw_samples[donor_span["sample_idx"]]
                    injected = self._inject(donor_raw, donor_span, hn)
                    if injected is not None:
                        augmented.append(injected)

            # Strategy 3: Concatenate concept-rich sentence with hard-neg sentence
            concept_rich_pool = [
                cp for cp in concept_spans
                if cp["sample_idx"] != hn["sample_idx"]
            ]
            if concept_rich_pool:
                picks = random.sample(
                    concept_rich_pool,
                    min(self.concat_per_hard_neg, len(concept_rich_pool)),
                )
                for cp in picks:
                    cat = self._concatenate(
                        raw_samples[cp["sample_idx"]], orig_sample
                    )
                    if cat is not None:
                        augmented.append(cat)

        logger.info(
            "HardNegativeMiner: generated %d augmented samples", len(augmented)
        )
        return augmented

    # =========================================================================
    # Embedding-based hard negative detection
    # =========================================================================

    def _find_hard_negatives(
        self, o_spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Return the subset of *o_spans* whose embedding is close to at least
        one concept anchor (cosine ≥ ``self.sim_threshold``), sorted by
        descending similarity and capped at ``self.max_hard_negatives``.
        """
        if not o_spans or self._embed_model is None or self._anchor_embs is None:
            # Fallback: return a deterministic random subset as pseudo-hard-negatives
            logger.warning(
                "HardNegativeMiner: embeddings unavailable – using length-based "
                "heuristic for hard negative selection."
            )
            return self._heuristic_hard_negatives(o_spans)

        texts = [s["text"] for s in o_spans]
        try:
            embs = self._embed_model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
        except Exception as exc:
            logger.warning("HardNegativeMiner: encoding failed (%s), using heuristic", exc)
            return self._heuristic_hard_negatives(o_spans)

        # Normalise for cosine similarity via matmul
        emb_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        anc_norm = self._anchor_embs / (
            np.linalg.norm(self._anchor_embs, axis=1, keepdims=True) + 1e-8
        )
        # (n_spans, n_anchors)
        sims: np.ndarray = emb_norm @ anc_norm.T
        max_sims: np.ndarray = sims.max(axis=1)

        scored = [
            {**o_spans[i], "anchor_sim": float(max_sims[i])}
            for i in range(len(o_spans))
            if float(max_sims[i]) >= self.sim_threshold
        ]
        scored.sort(key=lambda x: -x["anchor_sim"])
        return scored[: self.max_hard_negatives]

    def _heuristic_hard_negatives(
        self, o_spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        When embeddings are unavailable, approximate hard negatives by
        selecting longer O-spans (≥ 4 characters) since length correlates
        with noun-like form in Sinhala.
        """
        longer = [s for s in o_spans if len(s["text"]) >= 4]
        random.shuffle(longer)
        cap = min(self.max_hard_negatives, len(longer))
        return longer[:cap]

    # =========================================================================
    # Augmentation helpers
    # =========================================================================

    def _build_tagged_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a raw JSONL sample into the ``{'text', 'tags'}`` format
        expected by `load_annotated_data` consumers.
        """
        text: str = raw_sample["text"]
        tags: List[str] = ["O"] * len(text)
        for start, end, lbl in raw_sample.get("labels", []):
            if lbl == "concept":
                if start < len(tags):
                    tags[start] = "B-concept"
            elif lbl == "multiword-concept":
                if start < len(tags):
                    tags[start] = "B-multiword-concept"
        return {"text": text, "tags": tags}

    def _inject(
        self,
        donor_raw: Dict[str, Any],
        donor_span: Dict[str, Any],
        hard_neg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Strategy 2 – replace a real concept span in *donor_raw* with the
        hard-negative phrase, tagging the replacement position as O.

        This creates an explicit training signal: "this phrase, in this
        context, is NOT a concept."

        Parameters
        ----------
        donor_raw   : source sample whose concept span is replaced
        donor_span  : the specific concept span to swap out
        hard_neg    : hard-negative span dict (text, start, end, sample_idx)
        """
        orig_text: str = donor_raw["text"]
        hn_text: str = hard_neg["text"]
        span_s: int = donor_span["start"]
        span_e: int = donor_span["end"]

        # Safety: span must still fit in text (could have been from a different
        # version of the raw data)
        if span_s < 0 or span_e > len(orig_text) or span_s >= span_e:
            return None

        new_text = orig_text[:span_s] + hn_text + orig_text[span_e:]
        delta = len(hn_text) - (span_e - span_s)

        # Build character-level tag list — all O by default
        new_tags: List[str] = ["O"] * len(new_text)

        # Port concept labels from donor, adjusting offsets and skipping the
        # replaced span.
        for s, e, lbl in donor_raw.get("labels", []):
            if e <= span_s:
                # Before replaced span: no offset change
                new_s = s
            elif s >= span_e:
                # After replaced span: shift by delta
                new_s = s + delta
            else:
                # Overlaps the replaced range — skip (we replaced this concept)
                continue

            if 0 <= new_s < len(new_text):
                if lbl == "concept":
                    new_tags[new_s] = "B-concept"
                elif lbl == "multiword-concept":
                    new_tags[new_s] = "B-multiword-concept"
            # The injected hard_neg position stays O (no tag written)

        return {"text": new_text, "tags": new_tags}

    def _concatenate(
        self,
        concept_sample: Dict[str, Any],
        hard_neg_sample: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Strategy 3 – join a concept-rich sentence with the hard-negative
        sentence into a single training example.

        The combined text has the concept labels from *concept_sample*
        ported as-is, and the labels from *hard_neg_sample* ported after
        an offset shift.  The hard-negative span itself is already O in
        *hard_neg_sample* (by definition), so no special handling is needed
        for it.

        A full-stop separator is inserted between the two texts.
        """
        c_text: str = concept_sample["text"]
        h_text: str = hard_neg_sample["text"]

        # Avoid producing excessively long sequences that would be truncated
        if len(c_text) + len(h_text) > 450:
            return None

        sep = "."
        new_text = c_text + sep + " " + h_text
        offset = len(c_text) + len(sep) + 1  # +1 for the space

        new_tags: List[str] = ["O"] * len(new_text)

        # Port concept labels from the first sample unchanged
        for s, e, lbl in concept_sample.get("labels", []):
            if 0 <= s < len(new_text):
                if lbl == "concept":
                    new_tags[s] = "B-concept"
                elif lbl == "multiword-concept":
                    new_tags[s] = "B-multiword-concept"

        # Port labels from the second sample (shifted)
        for s, e, lbl in hard_neg_sample.get("labels", []):
            new_s = s + offset
            if 0 <= new_s < len(new_text):
                if lbl == "concept":
                    new_tags[new_s] = "B-concept"
                elif lbl == "multiword-concept":
                    new_tags[new_s] = "B-multiword-concept"

        return {"text": new_text, "tags": new_tags}

    # =========================================================================
    # Embedding model initialisation
    # =========================================================================

    def _load_embeddings(self) -> None:
        """Load sentence-transformers model and pre-compute anchor embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self._EMBED_MODEL)
            anc_embs = self._embed_model.encode(
                _CONCEPT_ANCHORS,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._anchor_embs = anc_embs.astype(np.float32)
            logger.info(
                "HardNegativeMiner: loaded embedding model '%s', "
                "%d anchor embeddings",
                self._EMBED_MODEL,
                len(_CONCEPT_ANCHORS),
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "HardNegativeMiner will use heuristic fallback."
            )
        except Exception as exc:
            logger.warning(
                "HardNegativeMiner: could not load embedding model (%s); "
                "using heuristic fallback.",
                exc,
            )


# ---------------------------------------------------------------------------
# Convenience function for use in training scripts
# ---------------------------------------------------------------------------

def load_and_augment(
    jsonl_path: str,
    *,
    sim_threshold: float = 0.55,
    max_hard_negatives: int = 150,
    upsample_factor: int = 1,
    inject_per_hard_neg: int = 2,
    concat_per_hard_neg: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    One-liner wrapper: load the training JSONL, mine hard negatives, and
    return the FULL augmented sample list (originals + synthetics).

    Typical usage::

        samples = load_and_augment("concept_train.jsonl")
        train_dataset = Dataset.from_list(samples)
    """
    # Load originals
    originals: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["text"]
            tags = ["O"] * len(text)
            for start, end, lbl in item.get("labels", []):
                if lbl == "concept" and start < len(tags):
                    tags[start] = "B-concept"
                elif lbl == "multiword-concept" and start < len(tags):
                    tags[start] = "B-multiword-concept"
            originals.append({"text": text, "tags": tags})

    miner = HardNegativeMiner(
        sim_threshold=sim_threshold,
        max_hard_negatives=max_hard_negatives,
        upsample_factor=upsample_factor,
        inject_per_hard_neg=inject_per_hard_neg,
        concat_per_hard_neg=concat_per_hard_neg,
        seed=seed,
    )
    synthetics = miner.mine_from_jsonl(jsonl_path)
    combined = originals + synthetics
    logger.info(
        "load_and_augment: %d originals + %d synthetics = %d total",
        len(originals),
        len(synthetics),
        len(combined),
    )
    return combined
