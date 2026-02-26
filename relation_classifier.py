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

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sinhala_morphology import get_morphology_handler, normalize_sinhala_word

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

_DIRECTIONALITY_CUES: Tuple[str, ...] = tuple(
    sorted(set([*_POST_ROLE.keys(), *_EFFECT_VERBS, *_AGENT_VERBS]), key=len, reverse=True)
)

_CLAUSE_BOUNDARY_MARKERS: Tuple[str, ...] = (
    ",",
    ";",
    ":",
    " - ",
    " — ",
    " නමුත් ",
    " එහෙත් ",
    " නම් ",
    " විට ",
)


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
# Feature dimensions (23 total):
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
#  [14] between_marker_density lexical intensity between entities
#  [15] morph_root_match       normalised root overlap flag
#  [16] morph_inflected_match  inflectional-variant flag
#  [17] e1_case_suffix         whether source entity ends in case marker
#  [18] e2_case_suffix         whether target entity ends in case marker
#  [19] between_connector      conjunction marker between entities
#  [20] cue_position           relative position of strongest directionality cue in between-span
#  [21] dependency_like_distance token/clause weighted proximity proxy
#  [22] clause_boundary        whether between-span crosses an internal clause boundary
# ---------------------------------------------------------------------------
_FEATURE_NAMES: List[str] = [
    "lex_is_a_between",
    "lex_part_between",
    "lex_cause_between",
    "lex_related_between",
    "lex_is_a_context",
    "lex_part_context",
    "lex_cause_context",
    "lex_related_context",
    "embed_sim",
    "between_is_a_sim",
    "between_part_sim",
    "between_cause_sim",
    "same_sentence",
    "distance_norm",
    "between_marker_density",
    "morph_root_match",
    "morph_inflected_match",
    "e1_case_suffix",
    "e2_case_suffix",
    "between_connector",
    "cue_position",
    "dependency_like_distance",
    "clause_boundary",
]

_WEIGHT_MATRIX = np.array([
    # is-a   part   cause  rel    is-a*  part*  cause* rel*   esim   b_is  b_pt  b_ca  ssent  dist  dens root infl e1cs e2cs conn  cuep  depd  clbd
    [  1.40,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00,  0.00,  0.00,  1.00, 0.00, 0.00, 0.30,  0.10, 0.20, 0.10, 0.00, 0.00, 0.00, 0.05, 0.02, 0.08, -0.05],
    [  0.00,  1.40,  0.00,  0.00,  0.00,  0.80,  0.00,  0.00,  0.00,  0.00, 1.00, 0.00, 0.30,  0.10, 0.15, 0.00, 0.00, 0.10, 0.20, 0.00, 0.00, 0.10, -0.04],
    [  0.00,  0.00,  1.40,  0.00,  0.00,  0.00,  0.90,  0.00,  0.00,  0.00, 0.00, 1.00, 0.30,  0.10, 0.20, 0.00, 0.00, 0.05, 0.05, 0.00, 0.12, 0.12, 0.06],
    [  0.00,  0.00,  0.00,  0.70,  0.00,  0.00,  0.00,  0.40,  0.50,  0.00, 0.00, 0.00, 0.40,  0.30, 0.10, 0.35, 0.30, 0.00, 0.00, 0.35, 0.00, 0.10, 0.04],
], dtype=np.float64)  # shape (4, 23)

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
_TOP_K_CONTEXTS_PER_PAIR: int = 3  # retain top-k contexts for each text-pair

_RELATION_MODEL_ENV = "RELATION_CLASSIFIER_MODEL_PATH"
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "relation_classifier_model.json"


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
        self._morph = get_morphology_handler()
        self._trained_model: Optional[Dict[str, Any]] = None
        self._init_prototype_embeddings()
        self._load_trained_model()
        # Base thresholds (will be adjusted adaptively)
        self._base_thresholds = {
            REL_IS_A: 0.42,
            REL_PART_OF: 0.48,
            REL_CAUSE: 0.45,
            REL_RELATED: 0.38,
        }
        self._domain_rel_offsets: Dict[str, Dict[str, float]] = {
            'education':   {REL_IS_A: -0.02, REL_PART_OF: -0.01, REL_CAUSE: 0.00, REL_RELATED: -0.02},
            'science':     {REL_IS_A: 0.00,  REL_PART_OF: 0.00,  REL_CAUSE: 0.01, REL_RELATED: 0.00},
            'environment': {REL_IS_A: 0.00,  REL_PART_OF: 0.01,  REL_CAUSE: 0.02, REL_RELATED: 0.00},
            'health':      {REL_IS_A: -0.01, REL_PART_OF: 0.00,  REL_CAUSE: 0.01, REL_RELATED: -0.01},
            'society':     {REL_IS_A: 0.00,  REL_PART_OF: 0.00,  REL_CAUSE: 0.01, REL_RELATED: 0.01},
        }
        self._noise_rel_offsets: Dict[str, Dict[str, float]] = {
            'clean':  {REL_IS_A: -0.02, REL_PART_OF: -0.01, REL_CAUSE: -0.01, REL_RELATED: -0.02},
            'medium': {REL_IS_A: 0.00,  REL_PART_OF: 0.00,  REL_CAUSE: 0.00,  REL_RELATED: 0.00},
            'noisy':  {REL_IS_A: 0.05,  REL_PART_OF: 0.04,  REL_CAUSE: 0.05,  REL_RELATED: 0.06},
        }

    def _resolve_model_path(self) -> Path:
        env_path = os.getenv(_RELATION_MODEL_ENV, "").strip()
        if env_path:
            return Path(env_path)
        return _DEFAULT_MODEL_PATH

    def _load_trained_model(self) -> None:
        path = self._resolve_model_path()
        if not path.exists():
            logger.info("RelationClassifier: no trained relation model found at %s; using fallback weights.", path)
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            feature_names = payload.get("feature_names", [])
            classes = payload.get("classes", [])
            coef = np.array(payload.get("coefficients", []), dtype=np.float64)
            intercept = np.array(payload.get("intercepts", []), dtype=np.float64)
            scaler_mean = np.array(payload.get("scaler_mean", []), dtype=np.float64)
            scaler_scale = np.array(payload.get("scaler_scale", []), dtype=np.float64)

            if feature_names != _FEATURE_NAMES:
                raise ValueError("feature_names mismatch between runtime and model artifact")
            if classes != CANONICAL_TYPES:
                raise ValueError("class ordering mismatch between runtime and model artifact")
            if coef.shape != (len(CANONICAL_TYPES), len(_FEATURE_NAMES)):
                raise ValueError(f"unexpected coefficient matrix shape: {coef.shape}")
            if intercept.shape != (len(CANONICAL_TYPES),):
                raise ValueError(f"unexpected intercept shape: {intercept.shape}")
            if scaler_mean.shape != (len(_FEATURE_NAMES),) or scaler_scale.shape != (len(_FEATURE_NAMES),):
                raise ValueError("invalid scaler statistics in model artifact")

            scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
            self._trained_model = {
                "coef": coef,
                "intercept": intercept,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "path": str(path),
                "metadata": payload.get("metadata", {}),
            }
            logger.info("RelationClassifier: loaded trained relation model from %s", path)
        except Exception as exc:
            self._trained_model = None
            logger.warning("RelationClassifier: failed loading model %s, using fallback weights. Error: %s", path, exc)

    def _predict_probs(self, feat: np.ndarray) -> np.ndarray:
        if self._trained_model is not None:
            mean = self._trained_model["scaler_mean"]
            scale = self._trained_model["scaler_scale"]
            z = (feat - mean) / scale
            logits = self._trained_model["coef"] @ z + self._trained_model["intercept"]
            return _softmax(logits)
        logits = _WEIGHT_MATRIX @ feat + _BIAS
        return _softmax(logits)

    def _calibrate_candidate_gates(self, domain: str, noise_bucket: str) -> Dict[str, float]:
        """Calibrate candidate pair generation gates by domain/noise bucket."""
        embed_gate = _EMBED_SIM_GATE
        ngram_gate = _NGRAM_SIM_GATE
        max_sent_distance = _MAX_SENT_DISTANCE
        max_offset_distance = _MAX_OFFSET_DISTANCE
        max_pairs_per_node = _MAX_PAIRS_PER_NODE

        if noise_bucket == 'clean':
            embed_gate -= 0.03
            ngram_gate -= 0.02
            max_sent_distance += 1
            max_offset_distance += 80
        elif noise_bucket == 'noisy':
            embed_gate += 0.07
            ngram_gate += 0.05
            max_sent_distance -= 1
            max_offset_distance -= 120
            max_pairs_per_node = max(3, max_pairs_per_node - 2)

        if domain in {'science', 'environment'}:
            embed_gate += 0.02
            ngram_gate += 0.01
        elif domain == 'education':
            embed_gate -= 0.01

        calibrated = {
            'embed_gate': max(0.25, min(0.70, embed_gate)),
            'ngram_gate': max(0.30, min(0.70, ngram_gate)),
            'max_sent_distance': int(max(1, min(3, max_sent_distance))),
            'max_offset_distance': int(max(320, min(900, max_offset_distance))),
            'max_pairs_per_node': int(max(3, min(8, max_pairs_per_node))),
        }
        return calibrated

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
        profile = self.engine.assess_text_bucket(text)
        domain = profile.get('domain', 'society')
        noise_bucket = profile.get('noise_bucket', 'medium')

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
            adjusted += self._domain_rel_offsets.get(domain, {}).get(rel_type, 0.0)
            adjusted += self._noise_rel_offsets.get(noise_bucket, {}).get(rel_type, 0.0)
            # Clamp to reasonable bounds
            adaptive_thresholds[rel_type] = max(0.25, min(0.65, adjusted))
        
        logger.debug(
            f"Adaptive thresholds: length={text_length}, entities={num_entities}, "
            f"density={entity_density:.2f}, quality={avg_entity_confidence:.2f}, "
            f"adjustment={adjustment:.2f}, domain={domain}, noise={noise_bucket}, thresholds={adaptive_thresholds}"
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
        profile = self.engine.assess_text_bucket(text)
        domain = profile.get('domain', 'society')
        noise_bucket = profile.get('noise_bucket', 'medium')
        candidate_gates = self._calibrate_candidate_gates(domain, noise_bucket)

        # ── Stage 1: candidate pairs ──────────────────────────────────────────
        candidates = self._generate_candidates(entities, sentences, text, candidate_gates)

        # ── Stage 2: classify each candidate ─────────────────────────────────
        # Keep all context-level predictions first, then aggregate by directed
        # relation key so confidence reflects multi-context evidence.
        aggregated_by_key: Dict[tuple, List[Dict[str, Any]]] = {}
        for pair in candidates:
            rel = self._classify_pair(pair, text)
            if rel is None:
                continue
            key = (rel["source"], rel["target"], rel["type"])
            aggregated_by_key.setdefault(key, []).append(rel)

        relationships: Dict[tuple, Dict[str, Any]] = {}
        for key, evidence in aggregated_by_key.items():
            merged = self._aggregate_relation_evidence(evidence)

            threshold = adaptive_thresholds.get(merged["type"], 0.45)
            if merged["confidence"] < threshold:
                continue

            rev_key = (merged["target"], merged["source"], merged["type"])
            if rev_key in relationships:
                existing = relationships[rev_key]
                # Prefer stronger directionality; break ties by confidence.
                if (
                    merged["direction_score"] > existing["direction_score"]
                    or (
                        merged["direction_score"] == existing["direction_score"]
                        and merged["confidence"] > existing["confidence"]
                    )
                ):
                    del relationships[rev_key]
                    relationships[key] = merged
            elif key not in relationships or relationships[key]["confidence"] < merged["confidence"]:
                relationships[key] = merged

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
        gates: Dict[str, float],
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

        pair_candidates: Dict[tuple, List[Dict[str, Any]]] = {}
        pair_count: Dict[str, int] = {}  # source key → count

        # Pre-compute embeddings for all entities (single batch call)
        entity_embs = self._batch_embed([e["text"] for e in entities])

        embed_gate = float(gates.get('embed_gate', _EMBED_SIM_GATE))
        ngram_gate = float(gates.get('ngram_gate', _NGRAM_SIM_GATE))
        max_sent_distance = int(gates.get('max_sent_distance', _MAX_SENT_DISTANCE))
        max_offset_distance = int(gates.get('max_offset_distance', _MAX_OFFSET_DISTANCE))
        max_pairs_per_node = int(gates.get('max_pairs_per_node', _MAX_PAIRS_PER_NODE))

        n = len(entities)
        for i in range(n):
            src_key = entities[i].get("normalized") or entities[i]["text"]

            for j in range(i + 1, n):
                if pair_count.get(src_key, 0) >= max_pairs_per_node:
                    break

                e1, e2 = entities[i], entities[j]
                if e1["text"] == e2["text"]:
                    continue

                pair_key = tuple(sorted([e1["text"], e2["text"]]))
                is_new_pair_for_src = pair_key not in pair_candidates
                if is_new_pair_for_src and pair_count.get(src_key, 0) >= max_pairs_per_node:
                    continue

                si, s1_text = sent_idx_of(e1)
                sj, s2_text = sent_idx_of(e2)

                same_sent = (si == sj and si >= 0)
                sent_dist = abs(si - sj) if si >= 0 and sj >= 0 else 99
                off_dist = abs(e1.get("offset", 0) - e2.get("offset", 0))

                # Gate 1: same sentence
                qualifies = same_sent
                # Gate 2: adjacent sentence
                if not qualifies and sent_dist <= max_sent_distance:
                    qualifies = True
                # Gate 3: embedding similarity
                if not qualifies and entity_embs is not None:
                    esim = self._cosine(entity_embs[i], entity_embs[j])
                    if esim >= embed_gate:
                        qualifies = True
                # Gate 4: character n-gram root similarity
                if not qualifies:
                    ng = _char_ngram_sim(e1["text"], e2["text"])
                    if ng >= ngram_gate:
                        qualifies = True

                if not qualifies:
                    continue
                if off_dist > max_offset_distance and not same_sent:
                    continue

                # Choose best context sentence
                context = s1_text if same_sent else (s1_text or s2_text)

                emb_sim = (
                    self._cosine(entity_embs[i], entity_embs[j])
                    if entity_embs is not None else 0.50
                )

                context_score = self._context_quality_score(
                    context=context,
                    same_sentence=same_sent,
                    sent_distance=sent_dist,
                    offset_distance=off_dist,
                    embed_sim=float(emb_sim),
                )

                if is_new_pair_for_src:
                    pair_count[src_key] = pair_count.get(src_key, 0) + 1

                pair_candidates.setdefault(pair_key, []).append({
                    "e1": e1, "e2": e2,
                    "context": context,
                    "same_sentence": same_sent,
                    "sent_distance": sent_dist,
                    "offset_distance": off_dist,
                    "embed_sim": float(emb_sim),
                    "context_score": context_score,
                    "e1_emb_idx": i,
                    "e2_emb_idx": j,
                    "_entity_embs": entity_embs,
                })

        candidates: List[Dict[str, Any]] = []
        for _, evidence in pair_candidates.items():
            ranked = sorted(
                evidence,
                key=lambda c: (
                    c.get("context_score", 0.0),
                    1.0 if c.get("same_sentence") else 0.0,
                    c.get("embed_sim", 0.0),
                ),
                reverse=True,
            )
            candidates.extend(ranked[:_TOP_K_CONTEXTS_PER_PAIR])

        return candidates

    def _context_quality_score(
        self,
        context: str,
        same_sentence: bool,
        sent_distance: int,
        offset_distance: int,
        embed_sim: float,
    ) -> float:
        """Heuristic ranking score used to retain top-k contexts per pair."""
        marker_hits = sum(1 for markers in _MARKERS.values() for marker in markers if marker in context)
        marker_signal = min(1.0, marker_hits / 4.0)

        same_sent_score = 1.0 if same_sentence else 0.0
        sent_proximity = max(0.0, 1.0 - (sent_distance / max(1, _MAX_SENT_DISTANCE + 1)))
        offset_proximity = max(0.0, 1.0 - (offset_distance / max(1, _MAX_OFFSET_DISTANCE)))

        return float(
            0.30 * same_sent_score
            + 0.25 * embed_sim
            + 0.20 * sent_proximity
            + 0.15 * offset_proximity
            + 0.10 * marker_signal
        )

    def _aggregate_relation_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple context-level predictions for the same relation key.

        Confidence uses noisy-or aggregation:
            1 - ∏(1 - p_i)
        """
        if len(evidence) == 1:
            rel = dict(evidence[0])
            rel["evidence_count"] = 1
            rel["contexts"] = [rel.get("context", "")]
            return rel

        best = max(evidence, key=lambda r: r.get("confidence", 0.0))

        confidences = [max(0.0, min(1.0, float(r.get("confidence", 0.0)))) for r in evidence]
        complement_prod = 1.0
        for c in confidences:
            complement_prod *= (1.0 - c)
        agg_conf = 1.0 - complement_prod

        score_keys = [REL_IS_A, REL_PART_OF, REL_CAUSE, REL_RELATED]
        agg_feature_scores: Dict[str, float] = {}
        for key in score_keys:
            vals = [float(r.get("feature_scores", {}).get(key, 0.0)) for r in evidence]
            agg_feature_scores[key] = round(sum(vals) / max(1, len(vals)), 4)

        contexts: List[str] = []
        for rel in evidence:
            ctx = str(rel.get("context", "")).strip()
            if ctx and ctx not in contexts:
                contexts.append(ctx)

        merged = dict(best)
        merged["confidence"] = round(float(min(1.0, agg_conf)), 4)
        merged["direction_score"] = round(float(max(r.get("direction_score", 0.0) for r in evidence)), 4)
        merged["directed"] = merged["direction_score"] > 0.60
        merged["feature_scores"] = agg_feature_scores
        merged["evidence_count"] = len(evidence)
        merged["contexts"] = contexts[:_TOP_K_CONTEXTS_PER_PAIR]
        return merged

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

        feat, between = self._extract_feature_vector(pair)
        probs = self._predict_probs(feat)

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
            "relation_family": "semantic_relatedness" if rel_type == REL_RELATED else "logical_relation",
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
            "model_source": "trained" if self._trained_model is not None else "fallback",
        }

    def _extract_feature_vector(self, pair: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        e1: Dict = pair["e1"]
        e2: Dict = pair["e2"]
        context: str = pair["context"]

        between = self._between_text(e1["text"], e2["text"], context)

        lex_between = self._marker_scores(between)
        lex_context = self._marker_scores(context)

        embed_sim = float(pair.get("embed_sim", 0.50))
        between_type_sims = self._between_type_similarities(between)

        same_sent = 1.0 if pair["same_sentence"] else 0.0
        dist_norm = max(0.0, 1.0 - pair["offset_distance"] / _MAX_OFFSET_DISTANCE)

        marker_hits_between = sum(1 for markers in _MARKERS.values() for m in markers if m in between)
        between_token_count = max(1, len(between.split()))
        marker_density = min(1.0, marker_hits_between / (between_token_count * 2.5))

        morph = self._morph_feature_vector(e1["text"], e2["text"], between)
        directionality = self._directionality_feature_vector(between)

        feat = np.array([
            lex_between[REL_IS_A],
            lex_between[REL_PART_OF],
            lex_between[REL_CAUSE],
            lex_between[REL_RELATED],
            lex_context[REL_IS_A],
            lex_context[REL_PART_OF],
            lex_context[REL_CAUSE],
            lex_context[REL_RELATED],
            embed_sim,
            between_type_sims[REL_IS_A],
            between_type_sims[REL_PART_OF],
            between_type_sims[REL_CAUSE],
            same_sent,
            dist_norm,
            marker_density,
            morph["root_match"],
            morph["inflected_match"],
            morph["e1_case_suffix"],
            morph["e2_case_suffix"],
            morph["between_connector"],
            directionality["cue_position"],
            directionality["dependency_like_distance"],
            directionality["clause_boundary"],
        ], dtype=np.float64)
        return feat, between

    def _morph_feature_vector(self, e1_text: str, e2_text: str, between: str) -> Dict[str, float]:
        root1 = normalize_sinhala_word(e1_text)
        root2 = normalize_sinhala_word(e2_text)
        root_match = 1.0 if root1 and root2 and root1 == root2 else 0.0

        inflected_match = 1.0 if self._morph.is_inflected_form(e1_text, e2_text) else 0.0

        case_suffixes = tuple(self._morph.case_suffixes)
        e1_case = 1.0 if any(e1_text.endswith(sfx) for sfx in case_suffixes) else 0.0
        e2_case = 1.0 if any(e2_text.endswith(sfx) for sfx in case_suffixes) else 0.0

        connector_tokens = ("සහ", "හා", "මෙන්ම", "සමඟ", "එක්ව")
        between_connector = 1.0 if any(tok in between for tok in connector_tokens) else 0.0

        return {
            "root_match": root_match,
            "inflected_match": inflected_match,
            "e1_case_suffix": e1_case,
            "e2_case_suffix": e2_case,
            "between_connector": between_connector,
        }

    def _directionality_feature_vector(self, between: str) -> Dict[str, float]:
        span = f" {between.strip()} " if between else " "
        span_clean = between.strip()

        cue_position = 0.5
        if span_clean:
            cue_hits: List[Tuple[int, int]] = []
            for cue in _DIRECTIONALITY_CUES:
                idx = span_clean.find(cue)
                if idx >= 0:
                    cue_hits.append((idx, len(cue)))
            if cue_hits:
                cue_idx, cue_len = min(cue_hits, key=lambda x: x[0])
                cue_center = cue_idx + (cue_len / 2.0)
                cue_position = float(np.clip(cue_center / max(1.0, float(len(span_clean))), 0.0, 1.0))

        clause_boundary = 1.0 if any(marker in span for marker in _CLAUSE_BOUNDARY_MARKERS) else 0.0
        clause_penalty = sum(1 for marker in _CLAUSE_BOUNDARY_MARKERS if marker in span)

        token_gap = len([tok for tok in re.split(r"\s+", span_clean) if tok])
        dependency_steps = token_gap + (2 * clause_penalty)
        dependency_like_distance = max(0.0, 1.0 - min(dependency_steps, 20) / 20.0)

        return {
            "cue_position": cue_position,
            "dependency_like_distance": dependency_like_distance,
            "clause_boundary": clause_boundary,
        }

    def build_training_samples(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        labels: Dict[Tuple[str, str], str],
        include_unlabeled_as_related: bool = True,
        max_unlabeled_ratio: float = 1.0,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Build feature vectors and labels from one annotated text instance.

        Parameters
        ----------
        text : str
            Source text.
        entities : List[Dict]
            Entity dicts containing at least ``text`` and ``offset``.
        labels : Dict[(source_text, target_text), relation_type]
            Directed relation labels.
        include_unlabeled_as_related : bool
            Whether unlabeled candidate pairs should be treated as ``related-to``.
        max_unlabeled_ratio : float
            Maximum unlabeled samples per labeled sample.
        """
        if len(entities) < 2:
            return [], []

        sentences = self.engine._split_sentences_with_spans(text)
        profile = self.engine.assess_text_bucket(text)
        candidate_gates = self._calibrate_candidate_gates(
            profile.get("domain", "society"),
            profile.get("noise_bucket", "medium"),
        )
        candidates = self._generate_candidates(entities, sentences, text, candidate_gates)

        X: List[np.ndarray] = []
        y: List[str] = []
        unlabeled_kept = 0
        labeled_count = 0

        for pair in candidates:
            e1 = pair["e1"]["text"]
            e2 = pair["e2"]["text"]

            label = labels.get((e1, e2))
            if label is None:
                label = labels.get((e2, e1))

            if label is None and include_unlabeled_as_related:
                allowed_unlabeled = max(1, int(max_unlabeled_ratio * max(labeled_count, 1)))
                if unlabeled_kept >= allowed_unlabeled:
                    continue
                label = REL_RELATED
                unlabeled_kept += 1

            if label is None:
                continue

            feat, _ = self._extract_feature_vector(pair)
            X.append(feat)
            y.append(label)
            if (e1, e2) in labels or (e2, e1) in labels:
                labeled_count += 1

        return X, y

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
