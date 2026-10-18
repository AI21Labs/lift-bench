"""How a check's outcome becomes its score.

A check that ran scores VALID_PARSING_SCORE + (1 - VALID_PARSING_SCORE) * verdict, so a failed check scores the
floor rather than 0.0; a target that isn't in the answer, or input the metric can't parse, scores 0.0; a check that
can't run (bad arguments, an unknown instruction) is an evaluation error.
"""

import pytest

import lift_bench.hierarchical_drafting.metrics  # noqa: F401  (registers every metric)
from lift_bench.hierarchical_drafting.metrics.registry import METRIC_REGISTRY
from lift_bench.hierarchical_drafting.output_parsers import nested_html
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.with_output_parser import (
    VALID_PARSING_SCORE,
)

DOC = "# T\n\n## Solar\n\nhello there.\n"


async def _score(metric: str, text: str = DOC, parser_selector: list | None = None, **metric_args) -> float:
    parser = (
        {"parser_name": "nested_html", "parser_args": {"selector_lst": parser_selector, "document_type": "markdown"}}
        if parser_selector is not None
        else {}
    )
    return await METRIC_REGISTRY[metric]([{"role": "assistant", "content": text}], metric_args=metric_args, **parser)


async def test_a_passed_check_scores_one():
    assert await _score("case_type", "all lowercase", case_type="lower") == 1.0


async def test_a_failed_check_scores_the_floor():
    assert await _score("case_type", "Not Lowercase", case_type="lower") == VALID_PARSING_SCORE


async def test_a_target_missing_from_the_answer_scores_zero():
    assert await _score("case_type", parser_selector=[["h1", 0], ["h2", 5]], case_type="lower") == 0.0


async def test_input_the_metric_cannot_parse_scores_zero():
    assert await _score("case_type", "text", case_type="sentence") == 0.0


@pytest.mark.parametrize(
    ("metric", "metric_args"),
    [
        ("emoji_count", {"lower_bound": None, "upper_bound": None}),
        ("emoji_count", {"lower_bound": 3, "upper_bound": 1}),
        ("ifbench", {"instruction_id_list": ["no:such_instruction"], "kwargs": [{}]}),
        ("lexical_match", {"string": "x", "match_type": "CONTAINS", "should_match": True, "fuzzy_match": True}),
        ("lexical_match", {"string": "", "match_type": "contains", "should_match": True, "fuzzy_match": True}),
        (
            "lexical_count",
            {
                "string": "x",
                "lower_bound": None,
                "upper_bound": None,
                "fuzzy_match": True,
                "strict_word_boundaries": True,
            },
        ),
    ],
    ids=[
        "no-emoji-bounds",
        "reversed-emoji-bounds",
        "unknown-instruction",
        "uppercase-match-type",
        "empty-string",
        "no-count-bounds",
    ],
)
async def test_a_check_that_cannot_run_is_an_evaluation_error(metric, metric_args):
    with pytest.raises(EvaluationError):
        await _score(metric, "text", **metric_args)


async def test_partial_verdicts_scale_between_the_floor_and_one():
    instructions = ["punctuation:punctuation_exclamation", "keywords:palindrome"]
    score = await _score(
        "ifbench", "plain text only.", instruction_id_list=instructions, kwargs=[{}, {}], hard_aggregation=False
    )

    assert score == VALID_PARSING_SCORE + (1 - VALID_PARSING_SCORE) * 0.5


LEXICAL = {"string": "hello", "match_type": "contains", "should_match": True, "fuzzy_match": True}


@pytest.mark.parametrize(
    "metric_args",
    [{k: v for k, v in LEXICAL.items() if k != "fuzzy_match"}, {**LEXICAL, "strict_word_boundaries": True}],
    ids=["missing-argument", "unexpected-argument"],
)
async def test_arguments_must_match_the_metric_signature(metric_args):
    with pytest.raises(EvaluationError, match="invalid arguments"):
        await _score("lexical_match", "hello there", **metric_args)


async def test_argument_values_are_coerced_to_the_declared_types():
    assert await _score("lexical_match", "Hello", **{**LEXICAL, "fuzzy_match": "false"}) == VALID_PARSING_SCORE
    assert await _score("emoji_count", "a 😀", lower_bound="1", upper_bound="2") == 1.0


async def test_parser_settings_that_cannot_be_used_score_zero():
    assert await _score("case_type", parser_selector=[], case_type="lower") == 0.0


async def test_a_failure_while_searching_the_answer_scores_zero(monkeypatch):
    def fail(*_):
        raise RuntimeError("tree error")

    monkeypatch.setattr(nested_html, "_select", fail)

    assert await _score("case_type", parser_selector=[["h1", 0]], case_type="lower") == 0.0


async def test_an_unknown_parser_is_an_evaluation_error():
    with pytest.raises(EvaluationError, match="unknown output parser"):
        await METRIC_REGISTRY["case_type"](
            [{"role": "assistant", "content": DOC}], parser_name="no_such_parser", metric_args={"case_type": "lower"}
        )


async def test_a_crash_while_rendering_the_selected_text_is_an_evaluation_error(monkeypatch):
    def crash(_node):
        raise RuntimeError("renderer bug")

    monkeypatch.setattr(nested_html.NestedHTMLParser, "_extract_deep", staticmethod(crash))

    with pytest.raises(EvaluationError, match="crashed"):
        await _score("case_type", parser_selector=[["h1", 0], ["h2", 0]], case_type="lower")
