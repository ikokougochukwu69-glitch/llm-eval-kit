"""Command-line entry point.

Examples:
    llm-eval                                    # offline: metrics + significance
    llm-eval --judge-provider anthropic --judge-model claude-haiku-5-5
    llm-eval --judge-provider anthropic --judge-model claude-haiku-5-5 --pairwise
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from .data import DATA_DIR, ROOT, load_records, systems
from .judge import get_client, grade, pairwise_position_test
from .metrics import score_records
from .report import build_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate LLM outputs and the evaluators themselves.")
    p.add_argument("--questions", type=Path, default=DATA_DIR / "questions.jsonl")
    p.add_argument("--responses", type=Path, default=DATA_DIR / "responses.jsonl")
    p.add_argument("--judge-provider", choices=["anthropic", "openai"])
    p.add_argument("--judge-model")
    p.add_argument("--pairwise", action="store_true", help="Run the position-bias test (needs a judge)")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    if args.pairwise and not args.judge_provider:
        p.error("--pairwise needs --judge-provider")
    if args.judge_provider and not args.judge_model:
        p.error("--judge-model is required with --judge-provider")

    scored = score_records(load_records(args.questions, args.responses))

    judged = bool(args.judge_provider)
    pairwise_results = []
    if judged:
        client = get_client(args.judge_provider, args.judge_model)
        print(f"Grading {len(scored)} responses with {args.judge_model}...")
        for r in scored:
            r.update(grade(client, r))
        if args.pairwise:
            names = systems(scored)
            if len(names) != 2:
                p.error("--pairwise needs exactly two systems in the responses file")
            by_key = {(r["system"], r["id"]): r for r in scored}
            ids = sorted({r["id"] for r in scored})
            print(f"Running position-bias test on {len(ids)} pairs...")
            pairwise_results = [
                pairwise_position_test(client, by_key[(names[0], i)], by_key[(names[1], i)])
                for i in ids
            ]

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(scored, judged, pairwise_results,
                          f"{args.judge_provider}/{args.judge_model}" if judged else "")
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    fields = [k for k in scored[0] if k != "references"]
    with open(run_dir / "scored.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    print(report)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
