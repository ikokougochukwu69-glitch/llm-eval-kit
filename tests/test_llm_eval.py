import pytest

from llm_eval.agreement import agreement_report, cohens_kappa
from llm_eval.data import load_records
from llm_eval.judge import grade, pairwise_position_test, parse_verdict, parse_winner
from llm_eval.metrics import contains_answer, exact_match, normalize, token_f1
from llm_eval.stats import paired_bootstrap


# ---------- metrics ----------

def test_normalize_strips_case_punctuation_articles():
    assert normalize("The  Nile!") == "nile"


def test_exact_match_handles_aliases():
    assert exact_match("Pacific", ["Pacific Ocean", "Pacific"]) == 1.0
    assert exact_match("It is the Pacific", ["Pacific"]) == 0.0


def test_contains_answer_is_whole_word():
    assert contains_answer("There are 206 bones", ["206"]) == 1.0
    assert contains_answer("There are 2060 bones", ["206"]) == 0.0


def test_contains_answer_false_positive_on_negation():
    # Known weakness this project demonstrates: mentions are not endorsements.
    assert contains_answer("It is not the Nile; it's the Congo.", ["Nile"]) == 1.0


def test_token_f1():
    assert token_f1("Albert Einstein", ["Albert Einstein"]) == 1.0
    assert token_f1("Isaac Newton", ["Albert Einstein"]) == 0.0
    assert 0 < token_f1("Einstein developed it", ["Albert Einstein"]) < 1


# ---------- agreement ----------

def test_kappa_perfect_and_chance():
    truth = [True, True, False, False]
    assert cohens_kappa(truth, truth) == pytest.approx(1.0)
    assert cohens_kappa([True] * 4, truth) == pytest.approx(0.0)


def test_agreement_report_counts():
    a = agreement_report([True, True, False, False], [True, False, True, False])
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (1, 1, 1, 1)
    assert a["precision"] == 0.5


# ---------- stats ----------

def test_bootstrap_detects_clear_difference():
    res = paired_bootstrap([1.0] * 30, [0.0] * 30, n_resamples=500)
    assert res["significant"] and res["difference"] == 1.0


def test_bootstrap_identical_systems_not_significant():
    scores = [1.0, 0.0] * 15
    assert not paired_bootstrap(scores, scores, n_resamples=500)["significant"]


def test_bootstrap_is_reproducible():
    a, b = [1, 0, 1, 1, 0] * 4, [0, 1, 1, 0, 0] * 4
    assert paired_bootstrap(a, b, n_resamples=300) == paired_bootstrap(a, b, n_resamples=300)


# ---------- judge (fake client, no API) ----------

class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, prompt):
        return self.replies.pop(0)


def test_parsers():
    assert parse_verdict("Looks right.\nVerdict: correct") is True
    assert parse_verdict("Verdict: Incorrect") is False
    assert parse_verdict("unsure") is None
    assert parse_winner("Winner: 2") == "2"
    assert parse_winner("Winner: tie") == "tie"


def _rec(system, response):
    return {"id": 1, "system": system, "question": "Q?", "references": ["X"], "response": response}


def test_grade_uses_verdict():
    out = grade(FakeClient(["Matches.\nVerdict: correct"]), _rec("a", "X"))
    assert out["judge_correct"] is True


def test_position_test_consistent_judge():
    # Prefers model_a in both orders: Response 1 first, then Response 2 after swap.
    res = pairwise_position_test(FakeClient(["Winner: 1", "Winner: 2"]),
                                 _rec("model_a", "X"), _rec("model_b", "Y"))
    assert res["consistent"] and res["pick_original_order"] == "model_a"


def test_position_test_detects_first_position_bias():
    res = pairwise_position_test(FakeClient(["Winner: 1", "Winner: 1"]),
                                 _rec("model_a", "X"), _rec("model_b", "Y"))
    assert not res["consistent"] and res["first_position_wins"] == 2


# ---------- bundled data ----------

def test_bundled_dataset_loads():
    records = load_records()
    assert len(records) == 48
    assert {r["system"] for r in records} == {"model_a", "model_b"}
