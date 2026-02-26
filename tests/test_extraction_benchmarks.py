import json
import os
from pathlib import Path
from statistics import mean

import pytest

from intelligent_mindmap_generator import IntelligentMindMapGenerator
from nlp_engine import SinhalaNLPEngine


ROOT = Path(__file__).resolve().parent.parent
GOLD_COMPAT = ROOT / "gold_data" / "sinhala_gold_train_compat.jsonl"
LABELED_60 = ROOT / "tests" / "data" / "sinhala_student_essays_labeled_60.jsonl"
REL_CONFUSION_OUT = ROOT / "tests" / "data" / "relation_confusion_matrix_latest.json"

REL_TYPES = ("is-a", "part-of", "cause-effect", "related-to")
REL_PRECISION_FLOORS = {
    "is-a": 0.05,
    "part-of": 0.05,
    "cause-effect": 0.01,
    "related-to": 0.04,
}
REL_BENCHMARK_MIN_CONFIDENCE = 0.60
REL_BENCHMARK_TOPK_PER_DOC = 24
REL_MIN_SUPPORT_FOR_STRICT_FLOOR = 10
REL_MATCH_THRESHOLDS = {
    "is-a": 0.45,
    "part-of": 0.35,
    "cause-effect": 0.35,
    "related-to": 0.45,
}


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _token_set(text: str):
    return {t for t in _normalize(text).split() if t}


def _string_similarity(a: str, b: str) -> float:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _count_fuzzy_matches(gold_items, pred_items, min_sim: float = 0.5):
    unmatched_pred = set(range(len(pred_items)))
    tp = 0
    for g in gold_items:
        best_idx = None
        best_sim = 0.0
        for i in list(unmatched_pred):
            sim = _string_similarity(g, pred_items[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx is not None and best_sim >= min_sim:
            tp += 1
            unmatched_pred.remove(best_idx)
    fp = len(unmatched_pred)
    fn = len(gold_items) - tp
    return tp, fp, fn


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _init_confusion_matrix():
    labels = ["__none__", *REL_TYPES]
    return {g: {p: 0 for p in labels} for g in labels}


def _safe_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def _precision_for_floor(tp: int, fp: int) -> float:
    """
    Precision used for floor gating.

    If a class has no predictions (tp+fp==0), we do not fail precision gates;
    coverage/recall is handled by other benchmark checks.
    """
    if (tp + fp) == 0:
        return 1.0
    return tp / (tp + fp)


def _pair_match_score(gs: str, gt: str, ps: str, pt: str) -> float:
    """Orientation-tolerant fuzzy score for relation pair alignment."""
    direct = 0.5 * _string_similarity(gs, ps) + 0.5 * _string_similarity(gt, pt)
    swapped = 0.5 * _string_similarity(gs, pt) + 0.5 * _string_similarity(gt, ps)
    return max(direct, swapped)


def _labels_to_entities(text: str, labels):
    entities = []
    for s, e, _ in labels:
        if 0 <= s < e <= len(text):
            entities.append(_normalize(text[s:e]))
    return {e for e in entities if e}


@pytest.fixture(scope="module")
def nlp_engine():
    os.environ.setdefault("DISABLE_SENTENCE_TRANSFORMERS", "true")
    return SinhalaNLPEngine()


@pytest.fixture(scope="module")
def generator():
    os.environ.setdefault("DISABLE_SENTENCE_TRANSFORMERS", "true")
    return IntelligentMindMapGenerator()


def test_entity_extraction_precision_recall_benchmark(nlp_engine):
    rows = _load_jsonl(GOLD_COMPAT)
    assert rows, "gold benchmark corpus missing"

    tp = fp = fn = 0
    for row in rows[:8]:
        text = row["text"]
        gold = _labels_to_entities(text, row.get("labels", []))
        pred_entities = nlp_engine.extract_entities(text)
        pred = [_normalize(e.get("text", "")) for e in pred_entities if e.get("text")]
        gold_list = list(gold)

        mtp, mfp, mfn = _count_fuzzy_matches(gold_list, pred, min_sim=0.45)
        tp += mtp
        fp += mfp
        fn += mfn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _f1(tp, fp, fn)

    assert precision >= 0.02
    assert recall >= 0.04
    assert f1 >= 0.03


def test_relationship_detection_accuracy(nlp_engine):
    rows = _load_jsonl(LABELED_60)
    assert len(rows) >= 50, "Need 50+ labeled essays for relationship benchmark"

    docs_with_rel = 0
    relation_f1_scores = []
    type_acc_scores = []
    confusion = _init_confusion_matrix()
    tp_by_type = {t: 0 for t in REL_TYPES}
    fp_by_type = {t: 0 for t in REL_TYPES}
    support_by_type = {t: 0 for t in REL_TYPES}

    for row in rows[:24]:
        text = row.get("text", "")
        expected = row.get("expected_relations", [])
        if not expected:
            continue

        docs_with_rel += 1
        pred_entities = nlp_engine.extract_entities(text)
        pred_relations = nlp_engine.extract_relationships(text, pred_entities)

        pred_candidates = [
            (
                _normalize(r.get("source", "")),
                _normalize(r.get("target", "")),
                r.get("type", ""),
                float(r.get("confidence", 0.0)),
            )
            for r in pred_relations
            if (
                r.get("source")
                and r.get("target")
                and r.get("type") in REL_TYPES
                and float(r.get("confidence", 0.0)) >= REL_BENCHMARK_MIN_CONFIDENCE
            )
        ]
        pred_candidates.sort(key=lambda x: x[3], reverse=True)
        pred_pairs = [(s, t, typ) for s, t, typ, _ in pred_candidates[:REL_BENCHMARK_TOPK_PER_DOC]]
        gold_pairs = [
            (_normalize(r.get("source_text", "")), _normalize(r.get("target_text", "")), r.get("type", ""))
            for r in expected
            if r.get("source_text") and r.get("target_text") and r.get("type") in REL_TYPES
        ]

        for _, _, gtype in gold_pairs:
            support_by_type[gtype] += 1

        unmatched_pred = set(range(len(pred_pairs)))
        unmatched_gold = set(range(len(gold_pairs)))
        tp = 0
        matched_type = 0
        for g_idx, (gs, gt, gtype) in enumerate(gold_pairs):
            best_idx = None
            best_score = 0.0
            for i in list(unmatched_pred):
                ps, pt, ptype = pred_pairs[i]
                score = _pair_match_score(gs, gt, ps, pt)
                if score > best_score:
                    best_score = score
                    best_idx = i
            match_threshold = REL_MATCH_THRESHOLDS.get(gtype, 0.45)
            if best_idx is not None and best_score >= match_threshold:
                tp += 1
                ps, pt, ptype = pred_pairs[best_idx]
                unmatched_gold.discard(g_idx)
                confusion[gtype][ptype] += 1
                if ptype == gtype:
                    matched_type += 1
                    tp_by_type[gtype] += 1
                else:
                    fp_by_type[ptype] += 1
                unmatched_pred.remove(best_idx)

        fp = len(unmatched_pred)
        fn = len(gold_pairs) - tp
        relation_f1_scores.append(_f1(tp, fp, fn))
        if tp > 0:
            type_acc_scores.append(matched_type / tp)

        for g_idx in unmatched_gold:
            _, _, gtype = gold_pairs[g_idx]
            confusion[gtype]["__none__"] += 1

        for p_idx in unmatched_pred:
            _, _, ptype = pred_pairs[p_idx]
            fp_by_type[ptype] += 1
            confusion["__none__"][ptype] += 1

    rel_metrics = {}
    for rtype in REL_TYPES:
        effective_floor = (
            REL_PRECISION_FLOORS[rtype]
            if support_by_type[rtype] >= REL_MIN_SUPPORT_FOR_STRICT_FLOOR
            else 0.0
        )
        rel_metrics[rtype] = {
            "support": support_by_type[rtype],
            "tp": tp_by_type[rtype],
            "fp": fp_by_type[rtype],
            "precision": round(_safe_precision(tp_by_type[rtype], fp_by_type[rtype]), 4),
            "precision_floor": REL_PRECISION_FLOORS[rtype],
            "effective_precision_floor": effective_floor,
        }

    REL_CONFUSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REL_CONFUSION_OUT, "w", encoding="utf-8") as out:
        json.dump(
            {
                "docs_evaluated": docs_with_rel,
                "relation_types": list(REL_TYPES),
                "metrics": rel_metrics,
                "confusion_matrix": confusion,
            },
            out,
            ensure_ascii=False,
            indent=2,
        )

    assert docs_with_rel >= 10
    assert mean(relation_f1_scores) >= 0.02
    assert (mean(type_acc_scores) if type_acc_scores else 0.0) >= 0.08

    for rtype, floor in REL_PRECISION_FLOORS.items():
        effective_floor = floor if support_by_type[rtype] >= REL_MIN_SUPPORT_FOR_STRICT_FLOOR else 0.0
        precision = _precision_for_floor(tp_by_type[rtype], fp_by_type[rtype])
        assert precision >= effective_floor, (
            f"precision floor failed for {rtype}: {precision:.4f} < {effective_floor:.4f} "
            f"(support={support_by_type[rtype]}, tp={tp_by_type[rtype]}, fp={fp_by_type[rtype]})"
        )


def test_graph_structure_quality_metrics_present(generator):
    text = (
        "පාසල් අධ්‍යාපනය ජාතික සංවර්ධනයට වැදගත් වේ. "
        "පළමුව ඉගෙනීම, ඊළඟට පුහුණුව, අවසානයේ ප්‍රයෝගය. "
        "සූර්ය ශක්තිය සහ ගල්අඟුරු අතර වෙනස ද වැදගත්ය."
    )
    out = generator.generate(text)
    meta = out.get("metadata", {})

    assert "quality" in meta
    assert "confidence_distribution" in meta
    assert "semantic_density" in meta
    assert "semantic_coverage" in meta
    assert "redundancy" in meta

    quality = meta["quality"]
    assert 0.0 <= quality.get("coherence_score", -1.0) <= 1.0
    assert 0.0 <= quality.get("semantic_coverage", -1.0) <= 1.0
    assert 0.0 <= quality.get("redundancy", -1.0) <= 1.0
    assert quality.get("semantic_density", -1.0) >= 0.0


def test_cross_validation_with_labeled_data(generator):
    rows = _load_jsonl(LABELED_60)[:24]
    assert len(rows) >= 20

    k = 3
    fold_size = len(rows) // k
    fold_scores = []

    for i in range(k):
        fold = rows[i * fold_size:(i + 1) * fold_size]
        if not fold:
            continue
        f1s = []
        for row in fold:
            gold = {_normalize(e) for e in row.get("expected_entities", []) if e}
            if not gold:
                continue
            pred = generator.generate(row["text"])
            pred_nodes = {
                _normalize(n.get("label", ""))
                for n in pred.get("nodes", [])
                if n.get("level", 0) > 0 and n.get("label")
            }
            tp = len(gold & pred_nodes)
            fp = len(pred_nodes - gold)
            fn = len(gold - pred_nodes)
            f1s.append(_f1(tp, fp, fn))
        if f1s:
            fold_scores.append(mean(f1s))

    assert len(fold_scores) >= 3
    assert mean(fold_scores) >= 0.04
