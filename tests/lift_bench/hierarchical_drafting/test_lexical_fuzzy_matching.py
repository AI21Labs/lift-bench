"""Fuzzy lexical matching also sees the answer's parsed markdown text."""

import pytest

import lift_bench.hierarchical_drafting.metrics  # noqa: F401  (registers every metric)
from lift_bench.hierarchical_drafting.metrics import _lexical_utils
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.registry import METRIC_REGISTRY
from lift_bench.hierarchical_drafting.metrics.with_output_parser import VALID_PARSING_SCORE as FAILED


async def _lexical_match(text: str, string: str, match_type: str = "contains", should_match: bool = True) -> bool:
    args = {"string": string, "match_type": match_type, "should_match": should_match, "fuzzy_match": True}
    return await METRIC_REGISTRY["lexical_match"]([{"role": "assistant", "content": text}], metric_args=args) == 1.0


async def _lexical_count(text: str, string: str, lower_bound: int | None, upper_bound: int | None) -> bool:
    args = {
        "string": string,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "fuzzy_match": True,
        "strict_word_boundaries": True,
    }
    return await METRIC_REGISTRY["lexical_count"]([{"role": "assistant", "content": text}], metric_args=args) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "**Machine** learning is great.",
        "Machine *learning* is great.",
        "__machine__ learning",
        "Machine\nlearning is great.",
        "Machine  learning is great.",
        "- machine\n  learning",
    ],
    ids=["bold", "italic", "underscore-bold", "line-wrap", "double-space", "wrapped-list-item"],
)
async def test_a_phrase_split_by_formatting_or_wrapping_still_matches(text):
    assert await _lexical_match(text, "machine learning")


async def test_a_forbidden_phrase_hidden_by_formatting_is_caught():
    assert not await _lexical_match("**Machine** learning", "machine learning", should_match=False)


@pytest.mark.parametrize(
    ("text", "string"),
    [("Machine\\-learning", "machine-learning"), ("R&amp;D budget", "R&D"), ("x **a**b", "ab")],
    ids=["escaped-hyphen", "html-entity", "adjacent-styles"],
)
async def test_escapes_and_entities_are_matched_as_rendered(text, string):
    assert await _lexical_match(text, string)


@pytest.mark.parametrize(
    ("text", "string", "match_type"),
    [
        ("**Introduction**", "Introduction", "exact"),
        ("# Intro\n\nbody", "body", "exact"),
        ("> Hello world", "hello", "starts_with"),
        ("*Hello* world", "hello", "starts_with"),
        ("Hello *world*", "world", "ends_with"),
    ],
)
async def test_anchored_match_types_see_through_markdown(text, string, match_type):
    assert await _lexical_match(text, string, match_type)


async def test_counts_come_from_the_first_text_variant_with_a_match():
    text = "**Machine** learning and **machine** learning"

    assert await _lexical_count(text, "machine learning", 1, 3)
    assert not await _lexical_count(text, "machine learning", None, 1)


async def test_without_fuzzy_matching_only_the_raw_text_counts():
    args = {"string": "machine learning", "match_type": "contains", "should_match": True, "fuzzy_match": False}
    score = await METRIC_REGISTRY["lexical_match"](
        [{"role": "assistant", "content": "**machine** learning"}], metric_args=args
    )

    assert score == FAILED


async def test_markdown_that_fails_to_parse_is_an_evaluation_error(monkeypatch):
    def fail(_markdown):
        raise ValueError("unparseable")

    monkeypatch.setattr(_lexical_utils, "parse_markdown", fail)

    with pytest.raises(EvaluationError):
        await _lexical_match("machine learning", "machine learning")
