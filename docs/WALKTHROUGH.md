# Walkthrough

A plain-language tour of the project. Read this before presenting or extending it.

## The core idea

An evaluation metric is a measuring instrument. Before trusting what it says about a model, you check the instrument against known answers. This project does that with ground-truth labels, then compares two models only through the evaluators that pass.

## The components

### `metrics.py`: three string metrics
- **normalize**: lowercase, remove punctuation and the words a/an/the. "The Nile!" becomes "nile". This is the standard SQuAD normalization.
- **exact_match**: is the normalized answer identical to a reference? Strict. "100 degrees Celsius" fails against "100".
- **contains_answer**: does a reference appear as a whole phrase inside the answer? Lenient. It can't tell "it is the Nile" from "it is **not** the Nile".
- **token_f1**: word overlap between answer and reference, as an F1 score. Long answers get low precision, so verbose correct answers score badly.

### `agreement.py`: grading the graders
For each evaluator, its correct/incorrect calls are compared with the ground-truth labels:
- **Precision**: when it says "correct", how often is it right?
- **Recall**: of the truly correct answers, how many does it catch?
- **Cohen's kappa**: agreement after removing what you'd get by chance. 1 = perfect, 0 = no better than guessing, negative = worse than guessing. Rough guide: >0.8 strong, 0.6–0.8 substantial, <0.4 weak.

### `stats.py`: paired bootstrap
To ask "is model A really better?": resample the 24 questions with replacement 10,000 times, recompute A − B each time, and take the middle 95% of results. If that range includes 0, the data can't separate the models. It's "paired" because each resample keeps both models' answers to the same question together.

### `judge.py`: LLM-as-judge
- **grade**: the judge gets the question, the reference and the answer, plus a rubric. It must end with `Verdict: correct` or `Verdict: incorrect`. The rubric specifically tells it that mentioning the reference while rejecting it counts as incorrect. That targets the failure contains_answer can't handle.
- **pairwise_position_test**: shows the judge both answers, asks which is better, then asks again with the order swapped. A fair judge picks the same *model* both times. A judge that picks "Response 1" both times prefers whatever comes first. This is a known, documented bias in LLM judges.

## How to read the bundled results

1. Exact match and F1 give the verbose model 0%. **Lesson:** string-overlap metrics measure format, not truth.
2. contains_answer has kappa 0.89 but still ranks the models in the wrong order, because both its mistakes fall on one model. **Lesson:** check *where* errors land, not just how many there are.
3. The bootstrap interval spans zero. **Lesson:** a 4-point gap on 24 questions means nothing.

## Questions you should be able to answer

- *Why kappa instead of accuracy?* Accuracy is inflated when one class dominates. Kappa isn't.
- *Why paired rather than independent bootstrap?* The same questions went to both models. Pairing removes variance from question difficulty and gives tighter, more honest intervals.
- *How would you scale this?* More questions (to narrow the CI), multiple human annotators (to measure inter-annotator agreement as the ceiling for any metric), and a graded rubric with weighted kappa.
- *How do you reduce judge position bias?* Run both orders and only count consistent verdicts, or average across orders.
- *Why is the judge also scored against ground truth?* Because it is a metric too, and it can be wrong.

## Extending it

- New metric: add a function to `METRICS` and a threshold to `THRESHOLDS` in `metrics.py`. It appears in every table automatically.
- New dataset: replace the two JSONL files (format in the README).
- New judge provider: write a class with `complete(prompt) -> str` and register it in `get_client`.
