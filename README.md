# llm-eval-kit

**Evaluating LLM outputs, and checking whether the evaluator can be trusted.**

Most LLM evaluation reports a score and stops there. This project asks the questions that come first:

1. **Does the metric agree with ground truth?** It measures precision, recall and Cohen's kappa for every evaluator against labeled answers.
2. **Is the gap between two models real?** It computes paired bootstrap confidence intervals instead of comparing raw percentages.
3. **Is the LLM judge biased?** It runs a pairwise test with answer order swapped, to catch position bias.

## Key findings (bundled dataset)

Two systems answer the same 24 factual questions. **model_a** gives terse answers. **model_b** gives verbose, natural sentences. Every response has a ground-truth correct/incorrect label.

| System | Ground truth | exact_match | contains_answer | token_f1 |
|---|---|---|---|---|
| model_a | 75% | 67% | 75% | 75% |
| model_b | 71% | **0%** | **79%** | **0%** |

| Evaluator | Precision | Recall | Cohen's kappa |
|---|---|---|---|
| contains_answer | 95% | 100% | 0.89 |
| token_f1 | 100% | 51% | 0.36 |
| exact_match | 100% | 46% | 0.31 |

What this shows:

- **Exact match and token F1 penalise style, not correctness.** They give the verbose model 0%, even though 71% of its answers are right. A team using these metrics would throw out a good model.
- **The best-agreeing metric still reverses the ranking.** contains_answer has the highest kappa (0.89), but both of its errors are false positives on model_b: *"written by Jane Austen's **sister**"* and *"It is **not** the Nile"*. A phrase being mentioned doesn't mean it's being asserted. As a result, the metric ranks model_b first when ground truth ranks model_a first. **High overall agreement does not guarantee a correct comparison.**
- **The 4-point gap between the models is noise.** The 95% paired bootstrap CI for the difference is [−25%, +33%]. With 24 questions, no ranking can be claimed in either direction.

> The responses and labels are a small, hand-constructed dataset. They were built to exercise known failure modes (paraphrase, verbosity, negation, near-miss entities), not sampled from real model output. The method transfers to real data: replace the two JSONL files with your own.

## Quick start

```bash
git clone https://github.com/ikokougochukwu69-glitch/llm-eval-kit.git
cd llm-eval-kit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest       # 15 tests, no API key needed
llm-eval     # metrics + significance, fully offline
```

### Add the LLM judge

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...              # Windows: set ANTHROPIC_API_KEY=...
llm-eval --judge-provider anthropic --judge-model claude-haiku-5-5 --pairwise
```

This adds an **LLM judge** column to both tables, so the judge is held to the same precision/recall/kappa standard as the string metrics. It also adds a **position-bias** section showing how often the judge picks the same system after the answers are swapped. OpenAI also works: `--judge-provider openai --judge-model <model>`.

Each run writes `results/<timestamp>/report.md` and `scored.csv`, which holds per-response scores and the judge's reasoning.

## How it works

```
data/questions.jsonl   question, reference answer, accepted aliases
data/responses.jsonl   system, response, ground-truth label
        │
        ▼
metrics.py     exact_match · contains_answer · token_f1  (SQuAD-style normalization)
judge.py       rubric grading · pairwise order-swap test
        │
        ▼
agreement.py   confusion matrix · precision · recall · Cohen's kappa vs ground truth
stats.py       paired bootstrap CI on the system difference
        │
        ▼
report.py      results/<timestamp>/report.md
```

## Design decisions

- **Kappa, not just accuracy.** When most answers are correct, an evaluator that always says "correct" scores high accuracy. Kappa corrects for chance agreement, so that evaluator scores 0.
- **Paired bootstrap.** Both systems answered the same questions, so the questions are resampled *together*. This removes question-difficulty variance from the comparison.
- **The judge is evaluated too.** An LLM judge is just another metric. It gets the same agreement table as the string metrics, and isn't treated as ground truth.
- **Temperature 0 and a fixed bootstrap seed** make runs reproducible.
- **Parse failures count as "not correct"**, never silently dropped. The raw judge output is kept in `scored.csv` for auditing.

## Using your own data

`questions.jsonl`:
```json
{"id": 1, "category": "geography", "question": "...", "reference": "...", "aliases": ["..."]}
```
`responses.jsonl` (one line per system per question):
```json
{"id": 1, "system": "my_model", "response": "...", "label": "correct"}
```

## Limitations

- Binary labels only. Partial credit would need a graded rubric and weighted kappa.
- The string metrics suit short factual answers. Open-ended generation needs the judge, or task-specific checks.
- 24 questions is enough to demonstrate the methods, not to rank real models.

## Project structure

```
llm-eval-kit/
├── data/                 questions + labeled responses
├── docs/WALKTHROUGH.md   explanation of every component
├── src/llm_eval/         metrics, judge, agreement, stats, report, cli
└── tests/                15 tests (judge tested with a fake client)
```

## License

MIT
