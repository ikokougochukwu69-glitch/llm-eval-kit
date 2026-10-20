"""Statistical comparison of two systems."""

import random


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap CI for mean(A) - mean(B), resampling questions with replacement.

    Paired: each resample keeps A's and B's answers to the same question
    together, which is the right design when both systems saw the same inputs.
    """
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("Score lists must be non-empty and the same length.")
    rng = random.Random(seed)
    n = len(scores_a)
    observed = sum(scores_a) / n - sum(scores_b) / n

    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(scores_a[i] - scores_b[i] for i in idx) / n)
    diffs.sort()

    alpha = (1 - confidence) / 2
    low = diffs[int(alpha * n_resamples)]
    high = diffs[int((1 - alpha) * n_resamples) - 1]
    return {
        "difference": observed,
        "ci_low": low,
        "ci_high": high,
        "significant": low > 0 or high < 0,
        "confidence": confidence,
    }
