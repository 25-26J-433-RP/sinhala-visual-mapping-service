import csv
import json
import re
from collections import Counter
from pathlib import Path


SINHALA_WORD = re.compile(r"[\u0D80-\u0DFF]{3,}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_entities(text: str, max_entities: int = 8):
    tokens = [t for t in SINHALA_WORD.findall(text) if len(t) >= 3]
    stop = {
        "එය", "මේ", "මෙම", "සහ", "හා", "නමුත්", "එහෙත්", "නිසා", "සඳහා", "ලෙස",
        "වේ", "විය", "ඇත", "කිරීම", "කිරීමට", "සිදු", "ගැන", "අතර", "තුළ", "මගින්",
    }
    tokens = [t for t in tokens if t not in stop]
    freq = Counter(tokens)
    ranked = [w for w, _ in freq.most_common(max_entities * 2)]
    entities = []
    seen = set()
    for token in ranked:
        if token in seen:
            continue
        idx = text.find(token)
        if idx < 0:
            continue
        entities.append({"text": token, "start": idx, "end": idx + len(token), "type": "concept"})
        seen.add(token)
        if len(entities) >= max_entities:
            break
    return entities


def _infer_relations(text: str, entities):
    if len(entities) < 2:
        return []
    relations = []
    cause_markers = ["නිසා", "හේතුවෙන්", "ප්‍රතිඵලයෙන්"]
    part_markers = ["කොටස", "අංග", "සංරචක", "ව්‍යුහ"]
    isa_markers = ["ලෙස", "යනු", "හැඳින්වේ"]

    for i in range(len(entities) - 1):
        src = entities[i]["text"]
        tgt = entities[i + 1]["text"]
        window = text[min(entities[i]["start"], entities[i + 1]["start"]): max(entities[i]["end"], entities[i + 1]["end"]) + 40]

        rel_type = "related-to"
        if any(m in window for m in cause_markers):
            rel_type = "cause-effect"
        elif any(m in window for m in part_markers):
            rel_type = "part-of"
        elif any(m in window for m in isa_markers):
            rel_type = "is-a"

        relations.append({
            "source_text": src,
            "target_text": tgt,
            "type": rel_type,
        })
    return relations


def build_corpus():
    root = Path(__file__).resolve().parent.parent
    input_csvs = [
        root.parent / "scoring-model-training" / "sinhala_dataset_final_with_dyslexic.csv",
        root.parent / "scoring-model-training" / "sinhala_dataset_final.csv",
        root.parent / "scoring-model-training" / "akura_dataset.csv",
    ]
    out_path = root / "tests" / "data" / "sinhala_student_essays_labeled_60.jsonl"

    rows = []
    for csv_path in input_csvs:
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = _normalize(row.get("essay_text") or row.get("input_text") or "")
                topic = _normalize(row.get("essay_topic") or row.get("topic") or "")
                if len(text) < 60:
                    continue
                rows.append({"text": text, "topic": topic})

    # deterministic dedup by text
    uniq = {}
    for r in rows:
        uniq.setdefault(r["text"], r)
    rows = list(uniq.values())[:60]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            entities = _extract_entities(row["text"], max_entities=8)
            relations = _infer_relations(row["text"], entities)
            record = {
                "id": f"student_{i:03d}",
                "source": "student_essay",
                "label_quality": "silver",
                "text": row["text"],
                "topic": row["topic"],
                "expected_entities": [e["text"] for e in entities],
                "expected_relations": relations,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} labeled essays -> {out_path}")


if __name__ == "__main__":
    build_corpus()
