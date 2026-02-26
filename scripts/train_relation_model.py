import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_engine import SinhalaNLPEngine
from relation_classifier import (
    CANONICAL_TYPES,
    REL_RELATED,
    _FEATURE_NAMES,
)
DEFAULT_OUTPUT = ROOT / "models" / "relation_classifier_model.json"
DEFAULT_INPUTS = [
    ROOT / "tests" / "data" / "sinhala_student_essays_human_reviewed_60.jsonl",
    ROOT / "gold_data" / "true_gold_v1" / "sinhala_true_gold_sentences_44.jsonl",
]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _find_offset(text: str, needle: str, start_from: int = 0) -> int:
    idx = text.find(needle, start_from)
    if idx >= 0:
        return idx
    return text.find(needle)


def _build_entities(text: str, raw_entities: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    entities: List[Dict[str, Any]] = []
    id_to_text: Dict[str, str] = {}
    cursor = 0

    for i, item in enumerate(raw_entities or [], start=1):
        if isinstance(item, str):
            ent_text = item.strip()
            ent_id = f"e{i}"
            start = _find_offset(text, ent_text, cursor)
        elif isinstance(item, dict):
            ent_text = str(item.get("text") or item.get("label") or item.get("value") or "").strip()
            ent_id = str(item.get("id") or f"e{i}")
            start = item.get("start")
            if start is None or not isinstance(start, int):
                start = _find_offset(text, ent_text, cursor)
        else:
            continue

        if not ent_text:
            continue
        if start is None or start < 0:
            continue

        cursor = start + len(ent_text)
        entities.append(
            {
                "id": ent_id,
                "text": ent_text,
                "offset": int(start),
                "confidence": 0.99,
                "type": "concept",
            }
        )
        id_to_text[ent_id] = ent_text

    return entities, id_to_text


def _build_labels(raw_relations: List[Any], id_to_text: Dict[str, str]) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], str] = {}

    for rel in raw_relations or []:
        if not isinstance(rel, dict):
            continue

        rel_type = str(rel.get("type") or "").strip()
        if rel_type not in CANONICAL_TYPES:
            continue

        src_text = str(rel.get("source_text") or "").strip()
        tgt_text = str(rel.get("target_text") or "").strip()

        if not src_text or not tgt_text:
            src_id = str(rel.get("source") or "").strip()
            tgt_id = str(rel.get("target") or "").strip()
            src_text = id_to_text.get(src_id, src_text)
            tgt_text = id_to_text.get(tgt_id, tgt_text)

        if not src_text or not tgt_text:
            continue

        labels[(src_text, tgt_text)] = rel_type

    return labels


def _load_training_records(inputs: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            continue
        records.extend(list(_iter_jsonl(path)))
    return records


def train_model(inputs: List[Path], output_path: Path, max_unlabeled_ratio: float) -> None:
    records = _load_training_records(inputs)
    if not records:
        raise RuntimeError("No training records found in the provided input files")

    engine = SinhalaNLPEngine()
    classifier = engine.relation_classifier

    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []

    for row in records:
        text = str(row.get("text") or "").strip()
        if len(text) < 10:
            continue

        raw_entities = row.get("expected_entities") or row.get("nodes") or []
        raw_relations = row.get("expected_relations") or row.get("relations") or []

        entities, id_to_text = _build_entities(text, raw_entities)
        if len(entities) < 2:
            continue

        labels = _build_labels(raw_relations, id_to_text)
        if not labels:
            continue

        x_part, y_part = classifier.build_training_samples(
            text=text,
            entities=entities,
            labels=labels,
            include_unlabeled_as_related=True,
            max_unlabeled_ratio=max_unlabeled_ratio,
        )
        X_rows.extend(x_part)
        y_rows.extend(y_part)

    if len(X_rows) < 12:
        raise RuntimeError(f"Insufficient training samples: {len(X_rows)}")

    class_counts = Counter(y_rows)
    if len(class_counts) < 2:
        raise RuntimeError(f"Need at least 2 classes to train logistic model. Found: {dict(class_counts)}")

    X = np.vstack(X_rows).astype(np.float64)
    y = np.array(y_rows)

    scaler_mean = X.mean(axis=0)
    scaler_scale = X.std(axis=0)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    X_scaled = (X - scaler_mean) / scaler_scale

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_scaled, y)

    coef = np.zeros((len(CANONICAL_TYPES), len(_FEATURE_NAMES)), dtype=np.float64)
    intercept = np.full((len(CANONICAL_TYPES),), -3.0, dtype=np.float64)

    trained_classes = list(model.classes_)
    for i, cls in enumerate(CANONICAL_TYPES):
        if cls in trained_classes:
            src_idx = trained_classes.index(cls)
            coef[i] = model.coef_[src_idx]
            intercept[i] = model.intercept_[src_idx]

    payload = {
        "version": "relation-logistic-v1",
        "feature_names": _FEATURE_NAMES,
        "classes": CANONICAL_TYPES,
        "coefficients": coef.tolist(),
        "intercepts": intercept.tolist(),
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "samples": int(len(X_rows)),
            "class_distribution": dict(class_counts),
            "inputs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs if p.exists()],
            "unlabeled_strategy": f"{REL_RELATED} with max ratio {max_unlabeled_ratio}",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Trained relation model on {len(X_rows)} samples")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Model written to: {output_path}")


def _default_existing_inputs() -> List[Path]:
    return [p for p in DEFAULT_INPUTS if p.exists()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic/linear relation classifier model from reviewed Sinhala annotations")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input JSONL path. Repeat for multiple files. If omitted, uses default reviewed/gold files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output model JSON file path",
    )
    parser.add_argument(
        "--max-unlabeled-ratio",
        type=float,
        default=1.0,
        help="Maximum unlabeled candidate ratio included as related-to",
    )
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.input] if args.input else _default_existing_inputs()
    if not input_paths:
        raise RuntimeError("No input files available. Pass --input with at least one reviewed/gold JSONL file.")

    train_model(
        inputs=input_paths,
        output_path=Path(args.output).resolve(),
        max_unlabeled_ratio=max(0.0, float(args.max_unlabeled_ratio)),
    )
