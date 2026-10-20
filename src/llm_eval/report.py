"""Build the markdown evaluation report."""

from .agreement import agreement_report
from .data import systems
from .metrics import METRICS, predict_correct
from .stats import paired_bootstrap


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def _num(x):
    return "n/a" if x is None else f"{x:.2f}"


def system_scores_table(scored: list[dict], judged: bool) -> str:
    cols = ["Ground truth", *METRICS] + (["LLM judge"] if judged else [])
    lines = ["| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name in systems(scored):
        rows = [r for r in scored if r["system"] == name]
        n = len(rows)
        cells = [_pct(sum(r["label"] == "correct" for r in rows) / n)]
        cells += [_pct(sum(predict_correct(m, r[m]) for r in rows) / n) for m in METRICS]
        if judged:
            cells.append(_pct(sum(bool(r.get("judge_correct")) for r in rows) / n))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluator_agreement(scored: list[dict], judged: bool) -> dict[str, dict]:
    actual = [r["label"] == "correct" for r in scored]
    out = {m: agreement_report([predict_correct(m, r[m]) for r in scored], actual)
           for m in METRICS}
    if judged:
        out["llm_judge"] = agreement_report(
            [bool(r.get("judge_correct")) for r in scored], actual)
    return out


def agreement_table(agreement: dict[str, dict]) -> str:
    lines = [
        "| Evaluator | Accuracy | Precision | Recall | Cohen's kappa | False positives | False negatives |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, a in sorted(agreement.items(), key=lambda kv: -kv[1]["kappa"]):
        lines.append(
            f"| {name} | {_pct(a['accuracy'])} | {_pct(a['precision'])} | "
            f"{_pct(a['recall'])} | {_num(a['kappa'])} | {a['fp']} | {a['fn']} |"
        )
    return "\n".join(lines)


def error_examples(scored: list[dict], metric: str, limit: int = 3) -> str:
    """Show cases where a metric disagrees with ground truth."""
    lines = []
    for r in scored:
        says = predict_correct(metric, r[metric])
        truth = r["label"] == "correct"
        if says != truth:
            kind = "false positive" if says else "false negative"
            lines.append(f"- *{kind}* ({r['system']}, Q{r['id']}): \"{r['response']}\"")
        if len(lines) == limit:
            break
    return "\n".join(lines) or "- none"


def significance_section(scored: list[dict]) -> str:
    names = systems(scored)
    if len(names) != 2:
        return "_Significance test needs exactly two systems._"
    a, b = names

    def correctness(name):
        rows = sorted((r for r in scored if r["system"] == name), key=lambda r: r["id"])
        return [float(r["label"] == "correct") for r in rows]

    res = paired_bootstrap(correctness(a), correctness(b))
    verdict = ("The difference **is** statistically significant."
               if res["significant"] else
               "The interval includes zero, so this data **cannot** say which system is better.")
    return (
        f"Ground-truth accuracy difference ({a} − {b}): **{res['difference']:+.1%}**  \n"
        f"{res['confidence']:.0%} paired bootstrap CI: [{res['ci_low']:+.1%}, {res['ci_high']:+.1%}]  \n"
        f"{verdict}"
    )


def position_bias_section(results: list[dict]) -> str:
    if not results:
        return "_Not run. Pass `--judge-provider` and `--pairwise` to enable._"
    n = len(results)
    consistent = sum(r["consistent"] for r in results)
    first_wins = sum(r["first_position_wins"] for r in results)
    decisions = 2 * n
    return (
        f"- Same system picked in both orders: **{consistent}/{n} ({consistent / n:.0%})**\n"
        f"- \"Response 1\" chosen in {first_wins}/{decisions} judgments "
        f"({first_wins / decisions:.0%}; 50% = no position preference)"
    )


def build_report(scored, judged, pairwise_results, judge_label="") -> str:
    agreement = evaluator_agreement(scored, judged)
    parts = [
        "# LLM Evaluation Report",
        f"\n{len(scored)} responses from {len(systems(scored))} systems."
        + (f" Judge: `{judge_label}`." if judged else " LLM judge not run."),
        "\n## 1. System scores by evaluator\n",
        "Each column is the share of answers that evaluator marked correct.\n",
        system_scores_table(scored, judged),
        "\n## 2. Which evaluator can we trust?\n",
        "Each evaluator compared with ground-truth labels, sorted by kappa.\n",
        agreement_table(agreement),
        "\n### Where the metrics go wrong\n",
    ]
    for m in METRICS:
        parts += [f"**{m}**\n", error_examples(scored, m), ""]
    parts += [
        "## 3. Is the difference between systems real?\n",
        significance_section(scored),
        "\n## 4. Judge position bias\n",
        position_bias_section(pairwise_results),
        "",
    ]
    return "\n".join(parts)
