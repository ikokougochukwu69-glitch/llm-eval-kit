"""Load questions and model responses, and join them into evaluation records."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
VALID_LABELS = {"correct", "incorrect"}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(
    questions_path: Path = DATA_DIR / "questions.jsonl",
    responses_path: Path = DATA_DIR / "responses.jsonl",
) -> list[dict]:
    """One record per (question, system) pair, with the question fields merged in."""
    questions = {q["id"]: q for q in read_jsonl(questions_path)}
    records = []
    for r in read_jsonl(responses_path):
        if r["id"] not in questions:
            raise ValueError(f"Response references unknown question id {r['id']}")
        if r["label"] not in VALID_LABELS:
            raise ValueError(f"Invalid label '{r['label']}' for question {r['id']}")
        q = questions[r["id"]]
        records.append({
            **r,
            "question": q["question"],
            "category": q["category"],
            "references": [q["reference"], *q.get("aliases", [])],
        })
    return records


def systems(records: list[dict]) -> list[str]:
    return sorted({r["system"] for r in records})
