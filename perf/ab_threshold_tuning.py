import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intelligent_mindmap_generator import IntelligentMindMapGenerator


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_threshold(generator: IntelligentMindMapGenerator, rows, rel_threshold: float):
    entity_scores = []
    quality_scores = []

    for row in rows:
        gold = {_normalize(e) for e in row.get("expected_entities", []) if e}
        if not gold:
            continue

        out = generator.generate(
            row["text"],
            options={"relationship_threshold": rel_threshold},
        )

        pred = {
            _normalize(n.get("label", ""))
            for n in out.get("nodes", [])
            if n.get("level", 0) > 0 and n.get("label")
        }

        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        entity_scores.append(_f1(tp, fp, fn))

        quality = out.get("metadata", {}).get("quality", {}).get("quality_score", 0.0)
        quality_scores.append(float(quality))

    return {
        "threshold": rel_threshold,
        "entity_f1": round(mean(entity_scores) if entity_scores else 0.0, 4),
        "quality_score": round(mean(quality_scores) if quality_scores else 0.0, 4),
        "combined": round(
            (0.65 * (mean(entity_scores) if entity_scores else 0.0))
            + (0.35 * (mean(quality_scores) if quality_scores else 0.0)),
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="A/B threshold tuning for Sinhala mindmap extraction")
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "tests" / "data" / "sinhala_student_essays_labeled_60.jsonl"),
        help="Path to labeled corpus JSONL",
    )
    parser.add_argument(
        "--thresholds",
        default="0.35,0.4,0.45,0.5,0.55,0.6",
        help="Comma-separated relationship thresholds",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum corpus samples for quick A/B run",
    )
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_SENTENCE_TRANSFORMERS", "true")

    corpus_path = Path(args.corpus)
    rows = load_jsonl(corpus_path)[: max(1, args.max_samples)]
    generator = IntelligentMindMapGenerator()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    results = [evaluate_threshold(generator, rows, t) for t in thresholds]
    results.sort(key=lambda r: r["combined"], reverse=True)

    print("A/B threshold tuning results")
    for r in results:
        print(
            f"threshold={r['threshold']:.2f} | entity_f1={r['entity_f1']:.4f} | "
            f"quality={r['quality_score']:.4f} | combined={r['combined']:.4f}"
        )

    if results:
        best = results[0]
        print(f"BEST threshold={best['threshold']:.2f}")


if __name__ == "__main__":
    main()
