"""LLM-as-judge: rubric grading and a pairwise position-bias test.

A client is any object with `complete(prompt: str) -> str`. Real clients for
Anthropic and OpenAI are provided; tests use a fake client.
"""

import os
import re

GRADE_PROMPT = """You are grading an answer to a factual question.

Question: {question}
Accepted reference answer(s): {references}
Answer to grade: {response}

Rubric:
- "correct": the answer states the reference fact (paraphrases and extra true detail are fine).
- "incorrect": the answer states a different fact, contradicts the reference, or only mentions the reference while rejecting it.
Judge factual content only. Ignore style and length.

Reply with one short sentence of reasoning, then a final line exactly in the form:
Verdict: correct
or
Verdict: incorrect"""

PAIRWISE_PROMPT = """Two assistants answered the same factual question.

Question: {question}
Accepted reference answer(s): {references}

Response 1: {first}
Response 2: {second}

Which response is better? Prioritise factual correctness, then concision.
Reply with one short sentence of reasoning, then a final line exactly in the form:
Winner: 1
or
Winner: 2
or
Winner: tie"""

_VERDICT = re.compile(r"verdict\s*:\s*(correct|incorrect)", re.IGNORECASE)
_WINNER = re.compile(r"winner\s*:\s*(1|2|tie)", re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    found = _VERDICT.findall(text or "")
    return found[-1].lower() == "correct" if found else None


def parse_winner(text: str) -> str | None:
    found = _WINNER.findall(text or "")
    return found[-1].lower() if found else None


def grade(client, record: dict) -> dict:
    prompt = GRADE_PROMPT.format(
        question=record["question"],
        references=" / ".join(record["references"]),
        response=record["response"],
    )
    output = client.complete(prompt)
    return {"judge_correct": parse_verdict(output), "judge_output": output.strip()}


def pairwise_position_test(client, rec_a: dict, rec_b: dict) -> dict:
    """Ask the judge twice with the order swapped.

    A position-robust judge picks the same *system* both times. If it picks
    "Response 1" both times, it is favouring whichever answer comes first.
    """
    refs = " / ".join(rec_a["references"])
    q = rec_a["question"]
    first = parse_winner(client.complete(PAIRWISE_PROMPT.format(
        question=q, references=refs, first=rec_a["response"], second=rec_b["response"])))
    second = parse_winner(client.complete(PAIRWISE_PROMPT.format(
        question=q, references=refs, first=rec_b["response"], second=rec_a["response"])))

    def to_system(winner, order):
        if winner in (None, "tie"):
            return winner
        return order[int(winner) - 1]

    pick_ab = to_system(first, (rec_a["system"], rec_b["system"]))
    pick_ba = to_system(second, (rec_b["system"], rec_a["system"]))
    return {
        "id": rec_a["id"],
        "pick_original_order": pick_ab,
        "pick_swapped_order": pick_ba,
        "consistent": pick_ab == pick_ba and pick_ab is not None,
        "first_position_wins": [first, second].count("1"),
    }


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 200):
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("Run: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("Set the ANTHROPIC_API_KEY environment variable.")
        self._client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIClient:
    def __init__(self, model: str):
        try:
            import openai
        except ImportError as e:
            raise SystemExit("Run: pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("Set the OPENAI_API_KEY environment variable.")
        self._client = openai.OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""


def get_client(provider: str, model: str):
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    raise SystemExit(f"Unknown judge provider '{provider}'. Use anthropic or openai.")
