"""Scoring matches AI21's internal implementation on a fixed set of tasks and completions.

`fixtures/scoring_parity_cases.jsonl` holds 100 tasks (5 from each published split's train and test files), a model's
completion for each, and the scores AI21's internal implementation gave: the task's aggregate and each check's score,
in `info.evaluations` order (None where the check, or the task, couldn't be scored).

The LLM judge isn't called: `judge_calls` maps the sha256 of every judge prompt the internal implementation sent to its
request settings and the judge's replies, which are played back here. A judge prompt that isn't in the recording means
the critic builds a different prompt, and is reported as such.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lift_bench.hierarchical_drafting.metrics  # noqa: F401  (registers every metric)
from lift_bench.hierarchical_drafting.metrics import critic
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.registry import constraint_reward
from lift_bench.hierarchical_drafting.utils.inference import CompletionResponse

CASES = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures" / "scoring_parity_cases.jsonl").read_text().splitlines()
]
TOLERANCE = 1e-9


class RecordedJudge:
    """Stands in for generate_completions in the critic, answering each prompt with its recorded replies."""

    def __init__(self, judge_calls: dict):
        self.judge_calls = judge_calls
        self.mismatches: list[str] = []

    async def __call__(self, prompts, temperature, max_tokens, n, **_kwargs):
        ((message,),) = prompts
        call = self.judge_calls.get(hashlib.sha256(message["content"].encode()).hexdigest())
        if call is None:
            self.mismatches.append(f"unrecorded judge prompt:\n{message['content'][:500]}")
            raise RuntimeError("unrecorded judge prompt")
        settings = {"temperature": temperature, "max_tokens": max_tokens, "n": n}
        recorded = {key: call[key] for key in settings}
        if settings != recorded:
            self.mismatches.append(f"judge request settings {settings}, recorded {recorded}")
        choices = [SimpleNamespace(message=SimpleNamespace(content=c)) for c in call["completions"]]
        return [CompletionResponse(choices=choices)]


async def _score(case: dict) -> tuple[list, float | None]:
    completion = [{"role": "assistant", "content": case["completion"]}]
    try:
        result = await constraint_reward(completion=completion, info=case["info"], prompt=case["prompt"])
    except EvaluationError as exc:
        return exc.agg_eval_scores, None
    return result["agg_eval_scores"], result["score"]


def _same(ours, expected) -> bool:
    return (ours is None and expected is None) or (
        ours is not None and expected is not None and abs(ours - expected) <= TOLERANCE
    )


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
async def test_scores_match_the_recorded_scores(case, monkeypatch):
    judge = RecordedJudge(case["judge_calls"])
    monkeypatch.setattr(critic, "generate_completions", judge)

    check_scores, score = await _score(case)

    evaluations = case["info"]["evaluations"]
    differences = [
        f"{e['evaluation_id']} ({e['metric_name']}): ours {ours}, expected {expected}"
        for e, ours, expected in zip(evaluations, check_scores, case["expected"]["check_scores"])
        if not _same(ours, expected)
    ]
    if not _same(score, case["expected"]["score"]):
        differences.append(f"task score: ours {score}, expected {case['expected']['score']}")
    assert not judge.mismatches and not differences, "\n".join(judge.mismatches + differences)
