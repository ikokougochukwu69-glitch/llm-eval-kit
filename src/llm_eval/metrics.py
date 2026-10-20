"""Reference-based metrics for short-answer QA.

Each metric returns a score in [0, 1]. `predict_correct` turns a score into a
correct / incorrect call so metrics can be compared against ground truth.
"""

import re
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(w for w in text.split() if w not in _ARTICLES)


def exact_match(response: str, references: list[str]) -> float:
    norm = normalize(response)
    return float(any(norm == normalize(ref) for ref in references))


def contains_answer(response: str, references: list[str]) -> float:
    """1.0 if any reference appears in the response as a whole-word phrase."""
    padded = f" {normalize(response)} "
    return float(any(f" {normalize(ref)} " in padded for ref in references))


def token_f1(response: str, references: list[str]) -> float:
    """Best token-overlap F1 against any reference."""
    pred = normalize(response).split()
    best = 0.0
    for ref in references:
        gold = normalize(ref).split()
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


METRICS = {
    "exact_match": exact_match,
    "contains_answer": contains_answer,
    "token_f1": token_f1,
}

# Score at or above which a metric "says" the answer is correct.
THRESHOLDS = {"exact_match": 1.0, "contains_answer": 1.0, "token_f1": 0.5}


def predict_correct(metric: str, score: float) -> bool:
    return score >= THRESHOLDS[metric]


def score_records(records: list[dict]) -> list[dict]:
    """Add every metric's score to each record (returns new dicts)."""
    scored = []
    for r in records:
        row = dict(r)
        for name, fn in METRICS.items():
            row[name] = fn(r["response"], r["references"])
        scored.append(row)
    return scored
