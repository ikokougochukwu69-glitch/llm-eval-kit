"""Measure how well an evaluator (metric or LLM judge) agrees with ground truth."""


def confusion(predicted: list[bool], actual: list[bool]) -> dict:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cohens_kappa(predicted: list[bool], actual: list[bool]) -> float:
    """Agreement corrected for chance. 1 = perfect, 0 = chance level, <0 = worse."""
    n = len(actual)
    if n == 0:
        return 0.0
    c = confusion(predicted, actual)
    observed = (c["tp"] + c["tn"]) / n
    p_yes = ((c["tp"] + c["fp"]) / n) * ((c["tp"] + c["fn"]) / n)
    p_no = ((c["fn"] + c["tn"]) / n) * ((c["fp"] + c["tn"]) / n)
    expected = p_yes + p_no
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_report(predicted: list[bool], actual: list[bool]) -> dict:
    c = confusion(predicted, actual)
    n = len(actual)

    def ratio(a, b):
        return a / b if b else None

    return {
        **c,
        "accuracy": ratio(c["tp"] + c["tn"], n),
        "precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "kappa": cohens_kappa(predicted, actual),
    }
