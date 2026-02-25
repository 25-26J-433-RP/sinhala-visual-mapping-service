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
# ENHANCED Lexical marker lists (EXPANDED from ~30 to ~90 markers)
# Includes student essay patterns for better classification accuracy
# ---------------------------------------------------------------------------
_MARKERS: Dict[str, List[str]] = {
    # IS-A markers: 9 → 30 markers (definitional, taxonomic, classification)
    REL_IS_A: [
        # Basic definitional markers
        "යනු", "හෙවත්", "කියන", "අර්ථය", "අර්ථයෙන්",
        "හැඳින්වෙයි", "ලෙස", "ලෙසින්",
        # Essay-style definitions
        "නිර්වචනය කළ හැක්කේ", "අර්ථ දක්වන්නේ", "හඳුන්වන්නේ",
        "ගණ්‍ය කරයි", "ශ්‍රේණිගත කරයි", "වර්ගය", "ප්‍රභේදය",
        # Exemplification (is-a through examples)
        "උදාහරණ", "උදාහරණයක්", "උදා:", "උදාහරණයක් ලෙස",
        "එනම්", "එබැවින්", "මෙහිදී", "මෙසේ",
        # Academic classification
        "ලෙස හඳුන්වන", "ලෙස නිර්වචනය කෙරේ", "ලෙස සැලකේ",
        "වේ", "වන", "වූ", "වන්නේ",
    ],
    
    # PART-OF markers: 8 → 20 markers (compositional, membership, inclusion)
    REL_PART_OF: [
        # Basic part/whole markers
        "අංශය", "කොටස", "අයත්", "අංග", "අභ්‍යන්තර",
        "භාගය", "ශාඛාව", "කොටසකි",
        # Extended compositional markers
        "කොටසක් වේ", "අයිති", "ඇතුළත්", "අඩංගු",
        "සමන්විත", "සමුදායේ", "කොටසක්", "දෙකොටස",
        # Membership and inclusion
        "අයිති වේ", "අයත් වේ", "තුළ පවතී", "ඇතුළත් වේ",
    ],
    
    # CAUSE-EFFECT markers: 9 → 25 markers (causal, functional, consequential)
    REL_CAUSE: [
        # Basic causal markers
        "නිසා", "හේතුවෙන්", "ප්‍රතිඵලයෙන්",
        "ප්‍රතිඵලයක් ලෙස", "ඒ හේතුවෙන්",
        # Action and causation
        "කරන", "ඇති කරයි", "සිදු කරයි",
        "ජනනය කරයි", "ඇතිවේ", "හේතුවක් වේ",
        # Essay-style causation
        "මේ හේතුව නිසා", "එම නිසා", "ඉන් ප්‍රතිඵලයක් ලෙස",
        "ප්‍රේරණය කරයි", "වටා ගැනීමට හේතුව", "බලපායි",
        # Consequence and result
        "ප්‍රතිඵලය", "ප්‍රතිඵලයන්", "ප්‍රතිඵල", "හේතුව",
        "ප්‍රතිඵලයක්", "හේතු", "හේතූන්", "බලපෑම",
        # Reason and explanation
        "හේතුවට", "හේතුවෙන් වශයෙන්", "හේතුවෙන්ම",
    ],
    
    # RELATED-TO markers: 8 → 15 markers (association, conjunction, proximity)
    REL_RELATED: [
        # Basic association markers
        "සහ", "හා", "මෙන්ම", "සමඟ", "ද", "සම්බන්ධ",
        "ආශ්‍රිත", "ආශ්‍රිතය",
        # Extended association
        "සම්බන්ධව", "සම්බන්ධයෙන්", "සමාන", "සමානව",
        "සමකාලීන", "එකට", "එක්ව",
    ],
}

# ---------------------------------------------------------------------------
# Directionality cue tables
# ---------------------------------------------------------------------------

# Postpositions / particles that reveal the role of the word that PRECEDES them.
# ENHANCED: Expanded from 23 to 60+ entries for better syntactic directionality
# Format:  token → role assigned to the preceding entity
#   'cause_src'   – preceding entity is the CAUSE   (source in cause-effect)
#   'cause_tgt'   – preceding entity is the EFFECT  (target in cause-effect)
#   'isa_child'   – preceding entity is the CHILD/SUBTYPE (source in is-a)
#   'part_part'   – preceding entity is the PART    (source in part-of)
#   'part_whole'  – preceding entity is the WHOLE   (target in part-of)
#   'agent'       – preceding entity is the AGENT   (source, any relation)
_POST_ROLE: Dict[str, str] = {
    # ── Causal cues (cause precedes cue) ──────────────────────────────────
    'නිසා':                 'cause_src',
    'හේතුවෙන්':             'cause_src',
    'ප්‍රතිඵලයෙන්':          'cause_src',
    'ඒ හේතුවෙන්':           'cause_src',
    'මගින්':                'cause_src',
    'හේතුවට':              'cause_src',
    'හේතුවෙන් වශයෙන්':      'cause_src',
    'නිසා වශයෙන්':          'cause_src',
    'බව නිසා':             'cause_src',
    'මුල්':                 'cause_src',
    'සලකා':                'cause_src',
    
    # ── Effect/result markers (preceding entity receives effect) ───────────
    'ලැබේ':                'cause_tgt',
    'වේ':                   'cause_tgt',
    'ඇතිවේ':               'cause_tgt',
    'සිදු වේ':              'cause_tgt',
    'ඇති වූ':               'cause_tgt',
    'සිදුවූ':               'cause_tgt',
    
    # ── Agent markers ──────────────────────────────────────────────────────
    'විසින්':               'agent',
    'විසින් කරන ලද':        'agent',
    'මගින්':                'agent',
    'තුළින්':               'agent',
    'හරහා':                'agent',
    
    # ── Definitional markers (child precedes cue) ──────────────────────────
    'යනු':                 'isa_child',
    'හෙවත්':               'isa_child',
    'ලෙස':                 'isa_child',
    'ලෙසින්':              'isa_child',
    'ලෙස හඳුන්වේ':         'isa_child',
    'ලෙස නිර්වචනය':        'isa_child',
    'ලෙස අර්ථ දක්වයි':     'isa_child',
    'ලෙස හඳුන්වන්නේ':      'isa_child',
    'ලෙස සැලකේ':           'isa_child',
    'වශයෙන්':              'isa_child',
    'වේ':                  'isa_child',
    'වන':                  'isa_child',
    'විට':                 'isa_child',
    
    # ── Genitive (preceding word is the WHOLE - target of part-of) ─────────
    'ගේ':                  'part_whole',
    'හි':                  'part_whole',
    'ල':                   'part_whole',
    'වල':                  'part_whole',
    'තුළ':                 'part_whole',
    'තුළ ඇති':             'part_whole',
    'අතර':                'part_whole',
    'අතරින්':              'part_whole',
    'සියල්ලෙහි':           'part_whole',
    'ඇතුලු':               'part_whole',
    'අන්තර්ගත':           'part_whole',
    
    # ── Part markers (preceding word is the PART - source of part-of) ──────
    'කොටසකි':              'part_part',
    'කොටසක්':             'part_part',
    'කොටස':               'part_part',
    'අංශය':               'part_part',
    'ශාඛාව':               'part_part',
    'අයිතිය':              'part_part',
    'අයත්':               'part_part',
    'අයත් වන':            'part_part',
    'අඩංගු':              'part_part',
    'ඇතුළත්':             'part_part',
    'භාගය':               'part_part',
    'දෙකොටස':             'part_part',
    'අංගයකි':             'part_part',
    'අපේක්ෂා':            'part_part',
}

# ENHANCED Verb/predicate patterns for better directionality resolution
# Verb/predicate patterns that, when they appear AFTER an entity, signal
# that the entity just BEFORE the verb is the EFFECT / product (target).
_EFFECT_VERBS: List[str] = [
    # Result verbs (entity receives the action/effect)
    'ඇතිවේ', 'ජනනය', 'නිෂ්පාදනය', 'ඇති කරයි', 'සිදු කරයි',
    'ලබා දෙයි', 'ලබා ගනී', 'කෙරේ', 'වේ', 'ඇති', 'ශාකය',
    # Extended result/state verbs
    'ලැබේ', 'ලැබුණි', 'සිදු වේ', 'සිදුවේ', 'සිදු වූ', 'සිදුවූ',
    'ඇතිවූ', 'ඇති වූ', 'ඇති වන', 'ඇතිවන', 'නිර්මාණය',
    'හටගනී', 'හටගත්', 'පැන නගී', 'බිහිවේ', 'විකසිත',
    # Essay-specific result verbs
    'දියුණු වේ', 'වර්ධනය වේ', 'වර්ධනය', 'පැවතුනි',
    'පැවතී', 'ඇති කෙරේ', 'බිහි කරයි', 'ජනිත',
]

# If an entity precedes one of these verbs, it is the AGENT/CAUSE (source).
_AGENT_VERBS: List[str] = [
    # Causative/transitive verbs (entity performs the action)
    'නිෂ්පාදනය කරයි', 'ජනනය කරයි', 'ඇති කරයි', 'ලබා දෙයි',
    'ඉදිරිපත් කරයි', 'නිපදවයි', 'ඇතිකරයි',
    # Extended causative verbs
    'ඇති කරන', 'සිදු කරන', 'සිදු කරයි', 'නිර්මාණය කරයි',
    'නිර්මාණය කරන', 'බලපායි', 'බලපාන', 'බලපෑම් කරයි',
    'ප්‍රේරණය කරයි', 'හේතු වෙයි', 'හේතු වන', 'හේතුකොට',
    # Essay-specific causative verbs
    'දියුණු කරයි', 'වර්ධනය කරයි', 'පෝෂණය කරයි',
    'සාර්ථක කරයි', 'ශක්තිමත් කරයි', 'වැඩිදියුණු කරයි',
    'මෙහෙයවයි', 'මඟ පෙන්වයි', 'යොමු කරයි',
]


class DirectionResolver:
    """
    Resolve the correct directed orientation (source → target) of a
    candidate relationship using Sinhala-specific syntax cues.

    Cue layers applied in priority order
    -------------------------------------
    1. **Strong postposition cues** – explicit role-marking particles
       (e.g. ``නිසා`` marks the preceding token as CAUSE).
    2. **Verb-pattern cues** – transitive verbs reveal agent (before verb)
       and patient/effect (after verb) using Sinhala SOV order.
    3. **Linear-order heuristic** – in the absence of syntactic evidence,
       the entity appearing earlier in the sentence is the source
       (Sinhala topic-comment structure puts the topic first).

    The resolver returns ``(source_text, target_text, direction_score)``
    where ``direction_score ∈ [0, 1]`` reflects how certain the orientation
    decision is (1.0 = explicit cue, 0.5 = fallback linear order).
    """

    # ── Type → applicable cue roles ──────────────────────────────────────
    _SRC_ROLES = {
        REL_IS_A:     {'isa_child', 'agent'},
        REL_PART_OF:  {'part_part'},
        REL_CAUSE:    {'cause_src', 'agent'},
        REL_RELATED:  set(),  # linear order only
    }
    _TGT_ROLES = {
        REL_IS_A:     set(),
        REL_PART_OF:  {'part_whole'},
        REL_CAUSE:    {'cause_tgt'},
        REL_RELATED:  set(),
    }

    def resolve(
        self,
        e1: Dict[str, Any],
        e2: Dict[str, Any],
        rel_type: str,
        sentence: str,
        between: str,
    ) -> Tuple[str, str, float]:
        """
        Return ``(source_text, target_text, direction_score)``.

        Parameters
        ----------
        e1, e2     : entity dicts from the hybrid extractor
        rel_type   : canonical relation type string
        sentence   : the context sentence containing both entities
        between    : text span lying between e1 and e2 in the sentence
        """
        t1, t2 = e1['text'], e2['text']
        off1 = e1.get('offset', 0)
        off2 = e2.get('offset', 0)

        # ── Layer 1: postposition scanning ───────────────────────────────
        role1 = self._postposition_role(t1, sentence)
        role2 = self._postposition_role(t2, sentence)

        src_roles = self._SRC_ROLES.get(rel_type, set())
        tgt_roles = self._TGT_ROLES.get(rel_type, set())

        # Case: e1 is explicitly marked as source
        if role1 in src_roles and role2 not in src_roles:
            return t1, t2, 0.90
        # Case: e2 is explicitly marked as source
        if role2 in src_roles and role1 not in src_roles:
            return t2, t1, 0.90
        # Case: e1 marked as target (whole in part-of)
        if role1 in tgt_roles and role2 not in tgt_roles:
            return t2, t1, 0.88
        # Case: e2 marked as target
        if role2 in tgt_roles and role1 not in tgt_roles:
            return t1, t2, 0.88

        # ── Layer 2: verb-pattern scanning ───────────────────────────────
        verb_src, verb_tgt, verb_score = self._verb_pattern_role(
            t1, t2, sentence, rel_type
        )
        if verb_src is not None:
            return verb_src, verb_tgt, verb_score

        # ── Layer 3: linear order (SOV topic-first heuristic) ────────────
        # Between-span cue: if a cause marker appears between e1 and e2,
        # the entity BEFORE the marker is the cause.
        cause_in_between = any(m in between for m in [
            'නිසා', 'හේතුවෙන්', 'ප්‍රතිඵලයෙන්', 'විසින්'
        ])
        if rel_type == REL_CAUSE and cause_in_between:
            # The entity that appears BEFORE the causal marker is the source
            src, tgt = (t1, t2) if off1 < off2 else (t2, t1)
            return src, tgt, 0.75

        # Default: earlier in text → source (topic-comment order)
        if off1 <= off2:
            return t1, t2, 0.50
        return t2, t1, 0.50

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _postposition_role(entity_text: str, sentence: str) -> Optional[str]:
        """
        Find the postposition token that immediately follows *entity_text*
        in *sentence* and return its assigned role, or None.

        Two passes are performed:

        1. Space-separated postpositions: scan up to 30 chars after the
           entity, strip leading whitespace, and check ``startswith``.
        2. Merged (suffix-attached) forms: Sinhala often agglutinates the
           postposition directly onto the noun without a space
           (e.g. "ගස" + "ේ" → "ගසේ").  Check whether *entity_text* +
           known suffix appears verbatim in the sentence.
        """
        idx = sentence.find(entity_text)
        if idx == -1:
            return None

        # Pass 1: space-separated lookup
        after = sentence[idx + len(entity_text): idx + len(entity_text) + 30].strip()
        for token, role in sorted(_POST_ROLE.items(), key=lambda x: -len(x[0])):
            if after.startswith(token):
                return role

        # Pass 2: merged suffix lookup (longest-first)
        # These are the phonological suffixes that directly agglutinate
        # onto the noun stem in Sinhala:
        #   ේ  / ගේ  → genitive (whole in part-of)
        #   හි         → locative/genitive (whole)
        #   ල          → genitive alternative
        #   නිසා       → causal (sometimes written without space)
        #   විසින්     → agentive
        _MERGED: List[Tuple[str, str]] = sorted([
            ("ගේ",     "part_whole"),
            ("නිසා",   "cause_src"),
            ("විසින්", "agent"),
            ("හේතුවෙන්", "cause_src"),
            ("හි",     "part_whole"),
            ("ල",      "part_whole"),
            ("ේ",      "part_whole"),
        ], key=lambda x: -len(x[0]))
        for suffix, role in _MERGED:
            if (entity_text + suffix) in sentence:
                return role

        return None

    @staticmethod
    def _verb_pattern_role(
        t1: str, t2: str, sentence: str, rel_type: str
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Use Sinhala SOV verb patterns to assign agent (source) and
        patient/effect (target).

        Returns ``(source, target, score)`` where source/target are text
        strings, or ``(None, None, 0.0)`` if no verb cue fired.
        """
        if rel_type not in (REL_CAUSE, REL_IS_A):
            return None, None, 0.0

        idx1 = sentence.find(t1)
        idx2 = sentence.find(t2)
        if idx1 == -1 or idx2 == -1:
            return None, None, 0.0

        # Agent verb: entity before this verb is the agent (source)
        for av in _AGENT_VERBS:
            av_idx = sentence.find(av)
            if av_idx == -1:
                continue
            # Whichever entity ends just before the verb is the agent
            end1 = idx1 + len(t1)
            end2 = idx2 + len(t2)
            if end1 <= av_idx and (end2 > av_idx or end2 < end1):
                return t1, t2, 0.80
            if end2 <= av_idx and (end1 > av_idx or end1 < end2):
                return t2, t1, 0.80

        # Effect verb: entity before the verb is the EFFECT (target);
        # the other entity is the CAUSE (source).
        for ev in _EFFECT_VERBS:
            ev_idx = sentence.find(ev)
            if ev_idx == -1:
                continue
            end1 = idx1 + len(t1)
            end2 = idx2 + len(t2)
            # Entity whose end is closest-before the effect verb is the EFFECT (target)
            gap1 = ev_idx - end1 if end1 <= ev_idx else 9999
            gap2 = ev_idx - end2 if end2 <= ev_idx else 9999
            if gap1 < gap2 and gap1 < 40:
                # t1 is effect (target), t2 is cause (source)
                return t2, t1, 0.78
            if gap2 < gap1 and gap2 < 40:
                # t2 is effect (target), t1 is cause (source)
                return t1, t2, 0.78

        return None, None, 0.0


# Module-level singleton constructed on first use inside RelationClassifier
_direction_resolver = DirectionResolver()


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
        # Base thresholds (will be adjusted adaptively)
        self._base_thresholds = {
            REL_IS_A: 0.42,
            REL_PART_OF: 0.48,
            REL_CAUSE: 0.45,
            REL_RELATED: 0.38,
        }

    def _calculate_adaptive_thresholds(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate adaptive confidence thresholds based on text characteristics.
        
        Adjusts thresholds lower for:
        - Longer, more complex essays (more context = more reliable)
        - Higher entity density (more concepts = clearer relationships)
        - Higher average entity confidence (better extraction quality)
        
        Adjusts thresholds higher for:
        - Short, simple texts (less context = less reliable)
        - Low entity density (sparse concepts)
        - Lower entity confidence (noisy extraction)
        
        Returns
        -------
        Dict[str, float]
            Per-type adjusted thresholds
        """
        # Text complexity metrics
        text_length = len(text)
        num_entities = len(entities)
        num_sentences = len(self.engine._split_sentences_with_spans(text))
        
        # Entity quality metrics
        avg_entity_confidence = sum(e.get("confidence", 0.5) for e in entities) / max(num_entities, 1)
        entity_density = num_entities / max(num_sentences, 1)  # entities per sentence
        
        # Normalization factors
        length_factor = min(1.0, text_length / 800)  # normalize to ~800 chars (typical essay paragraph)
        density_factor = min(1.0, entity_density / 4.0)  # normalize to 4 entities/sentence
        quality_factor = avg_entity_confidence  # already in [0,1]
        
        # Combined adjustment factor: higher = lower threshold (more permissive)
        # Formula: avg of three factors, weighted towards quality
        adjustment = (0.3 * length_factor + 0.3 * density_factor + 0.4 * quality_factor)
        
        # Apply adjustment to base thresholds
        # Adjustment range: -0.08 to +0.08 from base
        # Higher adjustment = lower threshold (subtract more)
        adaptive_thresholds = {}
        for rel_type, base in self._base_thresholds.items():
            # Adjust: reduce threshold when adjustment is high (good quality text)
            # Increase threshold when adjustment is low (poor quality text)
            adjusted = base - (adjustment - 0.5) * 0.16  # 0.16 = 2 * max_adjustment
            # Clamp to reasonable bounds
            adaptive_thresholds[rel_type] = max(0.25, min(0.65, adjusted))
        
        logger.debug(
            f"Adaptive thresholds: length={text_length}, entities={num_entities}, "
            f"density={entity_density:.2f}, quality={avg_entity_confidence:.2f}, "
            f"adjustment={adjustment:.2f}, thresholds={adaptive_thresholds}"
        )
        
        return adaptive_thresholds

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

        # ── Calculate adaptive thresholds based on text characteristics ───────
        adaptive_thresholds = self._calculate_adaptive_thresholds(text, entities)

        # ── Stage 1: candidate pairs ──────────────────────────────────────────
        candidates = self._generate_candidates(entities, sentences, text)

        # ── Stage 2: classify each candidate ─────────────────────────────────
        relationships: Dict[tuple, Dict[str, Any]] = {}
        for pair in candidates:
            rel = self._classify_pair(pair, text)
            if rel is None:
                continue
            # Use adaptive threshold instead of fixed threshold
            threshold = adaptive_thresholds.get(rel["type"], 0.45)
            if rel["confidence"] < threshold:
                continue
            # Use DIRECTED key: (source, target, type) — preserves orientation.
            # If the same undirected pair was seen with higher confidence,
            # keep that, but also allow both (A→B) and (B→A) if they survive
            # with different detected types.
            key = (rel["source"], rel["target"], rel["type"])
            rev_key = (rel["target"], rel["source"], rel["type"])
            # Prefer the orientation with the highest direction_score
            if rev_key in relationships:
                existing = relationships[rev_key]
                if rel["direction_score"] > existing["direction_score"]:
                    del relationships[rev_key]
                    relationships[key] = rel
                # else keep existing orientation
            elif key not in relationships or relationships[key]["confidence"] < rel["confidence"]:
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

        # ── Stage 3: Resolve directionality ─────────────────────────────────
        source, target, dir_score = _direction_resolver.resolve(
            e1, e2, rel_type, context, between
        )

        return {
            "source": source,
            "target": target,
            "type": rel_type,
            "confidence": round(confidence, 4),
            "direction_score": round(dir_score, 4),
            "directed": dir_score > 0.60,
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
