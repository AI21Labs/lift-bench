"""Sanity-check tests for the constraint_reward rubric.

Each test constructs a completion by hand where the expected per-metric
scores are known in advance, then asserts the weighted sum returned by
constraint_reward matches the expected value exactly.

Metric behaviour recap (a check scores 1.0 when it passes and FAILED, 0.00390625, when it fails):
  emoji_count   - 1.0 iff emoji count is within [lower_bound, upper_bound]
  lexical_match - 1.0 iff `string` is (or is not) found in the text
  case_type     - 1.0 iff every word satisfies the requested case
"""

import pytest

# Trigger @register_metric decorators before importing constraint_reward.
import lift_bench.hierarchical_drafting.metrics.case_type  # noqa: F401
import lift_bench.hierarchical_drafting.metrics.emoji_count  # noqa: F401
import lift_bench.hierarchical_drafting.metrics.lexical_match  # noqa: F401
from lift_bench.hierarchical_drafting.metrics.registry import constraint_reward
from lift_bench.hierarchical_drafting.metrics.with_output_parser import VALID_PARSING_SCORE as FAILED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(text: str) -> list[dict]:
    """Wrap plain text in the message-list format expected by all metrics."""
    return [{"role": "assistant", "content": text}]


def _info(*evaluations: dict) -> dict:
    return {"evaluations": list(evaluations)}


def _emoji_eval(weight: float, lower_bound: int | None = None, upper_bound: int | None = None) -> dict:
    return {
        "metric_name": "emoji_count",
        "weight": weight,
        "metric_args": {"lower_bound": lower_bound, "upper_bound": upper_bound},
    }


def _lexical_eval(weight: float, string: str, match_type: str = "contains", should_match: bool = True) -> dict:
    return {
        "metric_name": "lexical_match",
        "weight": weight,
        "metric_args": {"string": string, "match_type": match_type, "should_match": should_match, "fuzzy_match": True},
    }


def _case_eval(weight: float, case_type: str) -> dict:
    return {
        "metric_name": "case_type",
        "weight": weight,
        "metric_args": {"case_type": case_type},
    }


async def _score(completion, info) -> float:
    return (await constraint_reward(completion, info))["score"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_all_constraints_pass():
    """All three metrics satisfied -> total reward == 1.0."""
    # "hello world 😊🎉": 2 emojis, contains "hello", fully lowercase
    completion = _completion("hello world 😊🎉")
    info = _info(
        _emoji_eval(weight=0.5, lower_bound=1, upper_bound=3),  # 2 emojis  -> 1.0
        _lexical_eval(weight=0.3, string="hello"),  # found     -> 1.0
        _case_eval(weight=0.2, case_type="lower"),  # all lower -> 1.0
    )
    assert await _score(completion, info) == pytest.approx(1.0)


async def test_no_constraints_pass():
    """No metric satisfied -> total reward == FAILED."""
    # "Goodbye World": 0 emojis, no "hello", not lowercase
    completion = _completion("Goodbye World")
    info = _info(
        _emoji_eval(weight=0.5, lower_bound=1, upper_bound=3),  # 0 emojis -> FAILED
        _lexical_eval(weight=0.3, string="hello"),  # absent   -> FAILED
        _case_eval(weight=0.2, case_type="lower"),  # mixed    -> FAILED
    )
    assert await _score(completion, info) == pytest.approx(FAILED)


async def test_partial_constraints():
    """Only the emoji constraint passes (weight 0.5) -> total reward == 0.5 + 0.5 * FAILED."""
    # "great day 😊🎉": 2 emojis ✓, no "hello" ✗
    completion = _completion("great day 😊🎉")
    info = _info(
        _emoji_eval(weight=0.5, lower_bound=1, upper_bound=3),  # 2 emojis -> 1.0 x 0.5 = 0.5
        _lexical_eval(weight=0.5, string="hello"),  # absent   -> FAILED x 0.5
    )
    assert await _score(completion, info) == pytest.approx(0.5 + 0.5 * FAILED)


async def test_negated_lexical_match():
    """should_match=False: reward is 1.0 when the word is *absent*."""
    # Word "spam" must NOT appear
    completion = _completion("this is a clean response")
    info = _info(
        _lexical_eval(weight=1.0, string="spam", should_match=False),  # absent -> 1.0
    )
    assert await _score(completion, info) == pytest.approx(1.0)

    # Now it appears -> should fail
    completion_with_spam = _completion("this response contains spam")
    assert await _score(completion_with_spam, info) == pytest.approx(FAILED)


async def test_emoji_upper_bound_exceeded():
    """Too many emojis -> emoji_count fails even though lexical passes."""
    # 4 emojis exceeds upper_bound=3
    completion = _completion("hello 😊🎉🔥✨")
    info = _info(
        _emoji_eval(weight=0.5, lower_bound=1, upper_bound=3),  # 4 emojis -> FAILED x 0.5
        _lexical_eval(weight=0.5, string="hello"),  # found    -> 1.0 x 0.5 = 0.5
    )
    assert await _score(completion, info) == pytest.approx(0.5 + 0.5 * FAILED)


async def test_empty_evaluations_returns_one():
    """No evaluation specs -> default reward is 1.0."""
    assert await _score(_completion("anything"), {"evaluations": []}) == pytest.approx(1.0)


async def test_missing_evaluations_key_returns_one():
    """Missing `evaluations` key -> default reward is 1.0."""
    assert await _score(_completion("anything"), {}) == pytest.approx(1.0)


async def test_result_structure():
    """constraint_reward returns score, agg_eval_scores, and agg_eval_ids."""
    completion = _completion("hello world 😊🎉")
    info = _info(
        _emoji_eval(weight=0.5, lower_bound=1, upper_bound=3),
        _lexical_eval(weight=0.5, string="hello"),
    )
    result = await constraint_reward(completion, info)
    assert result["score"] == pytest.approx(1.0)
    assert result["agg_eval_scores"] == pytest.approx([1.0, 1.0])
    assert result["agg_eval_ids"] is None  # no metric_id keys in this info


async def test_agg_eval_ids_populated_when_evaluation_id_present():
    """agg_eval_ids is a list when any evaluation entry carries an evaluation_id."""
    completion = _completion("hello world 😊🎉")
    info = {
        "evaluations": [
            {**_emoji_eval(weight=0.5, lower_bound=1, upper_bound=3), "evaluation_id": "emoji-check"},
            _lexical_eval(weight=0.5, string="hello"),
        ]
    }
    result = await constraint_reward(completion, info)
    assert result["agg_eval_ids"] == ["emoji-check", None]
    assert result["agg_eval_scores"] == pytest.approx([1.0, 1.0])
