import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "tests" / "data" / "sinhala_student_essays_annotation_60.jsonl"
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "sinhala_student_essays_human_reviewed_60.jsonl"


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_valid_relation(rel: dict) -> bool:
    if not isinstance(rel, dict):
        return False
    return bool(rel.get("source_text")) and bool(rel.get("target_text")) and bool(rel.get("type"))


def promote(input_path: Path, output_path: Path) -> None:
    promoted = []
    skipped = 0

    for row in _iter_jsonl(input_path):
        status = row.get("annotation_status", "")
        expected_entities = row.get("expected_entities", [])
        expected_relations = row.get("expected_relations", [])

        if status != "human_verified":
            skipped += 1
            continue

        if not expected_entities or not isinstance(expected_entities, list):
            skipped += 1
            continue

        valid_relations = [r for r in expected_relations if _is_valid_relation(r)]

        promoted.append(
            {
                "id": row.get("id"),
                "source": row.get("source", "student_essay"),
                "label_quality": "human_gold",
                "text": row.get("text", ""),
                "topic": row.get("topic", ""),
                "expected_entities": expected_entities,
                "expected_relations": valid_relations,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for row in promoted:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Promoted {len(promoted)} reviewed records -> {output_path}")
    print(f"Skipped {skipped} records (not human_verified or invalid labels)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote human-verified annotation records to final labeled corpus")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input annotation JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output reviewed JSONL")
    args = parser.parse_args()

    promote(Path(args.input), Path(args.output))
