from typing import Any

import pytest

import lift_bench.hierarchical_drafting.metrics  # noqa: F401  (registers every metric)
from lift_bench.hierarchical_drafting.constraints.artifact import (
    ArtifactExistanceConstraint,
    ArtifactLLMJudgeConstraint,
)
from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.registry import (
    METRIC_REGISTRY,
    constraint_reward,
)
from lift_bench.hierarchical_drafting.metrics.with_output_parser import VALID_PARSING_SCORE as FAILED

DOC = "# Energy\n\nIntro.\n\n## Solar\n\nPanels convert sunlight.\n\n## Wind\n\nTurbines spin.\n"
SOLAR: dict[str, Any] = {
    "parser_name": "nested_html",
    "parser_args": {
        "selector_lst": [(HTMLTag.H1, 0), (HTMLTag.H2, 0)],
        "is_shallow": False,
        "document_type": "markdown",
    },
}
MISSING_SECTION = {**SOLAR, "parser_args": {**SOLAR["parser_args"], "selector_lst": [(HTMLTag.H1, 0), (HTMLTag.H2, 5)]}}
REQUEST = [{"role": "user", "content": "Write the 'Solar' section."}]


def _completion(text: str = DOC) -> list[dict]:
    return [{"role": "assistant", "content": text}]


async def _critic(**metric_args) -> float:
    args = {"critic_model_name": "judge", "criteria": ["Mentions sunlight."], "messages": REQUEST, **metric_args}
    return await METRIC_REGISTRY["light_absolute_critic"](_completion(), metric_args=args)


async def test_judge_sees_the_criterion_the_request_and_only_the_scoped_section(judge):
    evaluation = {
        "metric_name": "light_absolute_critic",
        "metric_args": {"critic_model_name": "judge", "criteria": ["Mentions sunlight."], "messages": REQUEST},
        **SOLAR,
    }

    await METRIC_REGISTRY["light_absolute_critic"](_completion(), **evaluation)

    model, prompt = judge.prompts[0]
    assert model == "judge"
    assert "Mentions sunlight." in prompt
    assert "Write the 'Solar' section." in prompt
    assert "Panels convert sunlight." in prompt
    assert "Turbines spin." not in prompt


@pytest.mark.parametrize(("decision", "expected"), [("1", 1.0), ("0", FAILED)])
async def test_binary_range_passes_on_true(judge, decision, expected):
    judge.decisions = [f"<decision>{decision}</decision>"]

    assert await _critic(use_binary_range=True) == expected


@pytest.mark.parametrize(("decision", "expected"), [("5", 1.0), ("4", 1.0), ("3", FAILED), ("1", FAILED)])
async def test_one_to_five_range_passes_at_the_threshold(judge, decision, expected):
    """Scores normalize to 0-1, so 4 -> 0.75 clears the 0.7 threshold and 3 -> 0.5 does not."""
    judge.decisions = [f"<decision>{decision}</decision>"]

    assert await _critic(use_binary_range=False) == expected
    assert "a score on a scale of 1 to 5" in judge.prompts[0][1]


async def test_out_of_range_scores_are_clamped_by_default(judge):
    judge.decisions = ["<decision>9</decision>"]

    assert await _critic(use_binary_range=False) == 1.0


async def test_out_of_range_scores_are_invalid_when_clamping_is_off(judge):
    judge.decisions = ["<decision>9</decision>"]

    with pytest.raises(EvaluationError):
        await _critic(use_binary_range=False, allow_score_out_of_range=False)


async def test_too_few_parseable_judgements_fail_the_evaluation(judge):
    judge.decisions = ["<decision>1</decision>", "no decision here", "<decision>x</decision>"]

    with pytest.raises(EvaluationError):
        await _critic(n=3)


async def test_min_score_threshold_requires_every_judgement_to_pass(judge):
    judge.decisions = ["<decision>1</decision>"] * 3 + ["<decision>0</decision>"]

    assert await _critic(n=4) == 1.0
    assert await _critic(n=4, threshold_key="min_score") == FAILED


async def test_judge_failure_is_an_evaluation_error(judge):
    judge.fail = True

    with pytest.raises(EvaluationError):
        await _critic()


async def test_without_any_request_the_judge_is_not_called(judge):
    with pytest.raises(EvaluationError):
        await METRIC_REGISTRY["light_absolute_critic"](
            _completion(), metric_args={"critic_model_name": "judge", "criteria": ["Mentions sunlight."]}
        )

    assert judge.prompts == []


async def test_the_rollout_prompt_is_the_fallback_request(judge):
    completion = [{"role": "user", "content": "Rollout prompt."}, *_completion()]

    await METRIC_REGISTRY["light_absolute_critic"](
        completion, metric_args={"critic_model_name": "judge", "criteria": ["Mentions sunlight."]}
    )

    assert "Rollout prompt." in judge.prompts[0][1]


async def test_request_can_be_withheld_from_the_judge(judge):
    await _critic(pass_prompt_to_llm_judge=False)

    prompt = judge.prompts[0][1]
    assert "<request>" not in prompt
    assert "Does the response adhere to the following instruction" in prompt


async def test_without_a_single_criterion_the_judge_rates_helpfulness(judge):
    await _critic(criteria=None)

    assert "helpful manner" in judge.prompts[0][1]


async def test_instructions_can_follow_the_response(judge):
    await _critic(instruction_after_completion=True)

    prompt = judge.prompts[0][1]
    assert prompt.index("</response>") < prompt.index("Does the response adhere")


@pytest.mark.parametrize(("parser", "expected"), [(SOLAR, 1.0), (MISSING_SECTION, 0.0)])
async def test_parser_evaluator_passes_only_when_the_section_exists(parser, expected):
    evaluation = {**ArtifactExistanceConstraint(criterion="", should_succeed=True).to_evaluation_dict(), **parser}

    assert await METRIC_REGISTRY["parser_evaluator"](_completion(), **evaluation) == expected


async def test_generated_judge_evaluations_score_end_to_end(judge):
    judge.decisions = ["<decision>5</decision>"]
    critic_constraint = ArtifactLLMJudgeConstraint(
        criterion="Mentions sunlight.",
        critic_model_name="judge",
        custom_user_prompt="Write the 'Solar' section.",
    )
    evaluations = [
        {**critic_constraint.to_evaluation_dict(), **SOLAR, "weight": 0.5, "evaluation_id": "content_0"},
        {
            **ArtifactExistanceConstraint(criterion="", should_succeed=True).to_evaluation_dict(),
            **MISSING_SECTION,
            "weight": 0.5,
            "evaluation_id": "structure_1",
        },
    ]

    result = await constraint_reward(_completion(), {"evaluations": evaluations})

    assert result["agg_eval_scores"] == [1.0, 0.0]
    assert result["score"] == 0.5


async def test_a_judge_check_without_messages_is_judged_against_the_task_prompt(judge):
    evaluation = {
        "metric_name": "light_absolute_critic",
        "metric_args": {"critic_model_name": "judge", "criteria": ["Mentions sunlight."]},
        **SOLAR,
        "weight": 1.0,
        "evaluation_id": "content_0",
    }

    await constraint_reward(_completion(), {"evaluations": [evaluation]}, prompt=REQUEST)

    assert "<request>\nWrite the 'Solar' section.\n</request>" in judge.prompts[0][1]


async def test_unregistered_metrics_are_reported_by_name():
    evaluation = {"metric_name": "absolute_critic", "metric_args": {}, "weight": 1.0, "evaluation_id": "content_0"}

    with pytest.raises(EvaluationError, match="'content_0' names unregistered metric 'absolute_critic'"):
        await constraint_reward(_completion(), {"evaluations": [evaluation]})


async def test_criterion_prompt_follows_the_template(judge):
    await _critic(use_binary_range=False)

    assert judge.prompts[0][1] == (
        "Denote the following request and response, enclosed by xml tags <request>REQUEST</request> and "
        "<response>RESPONSE</response> respectively.\n\n"
        "Based ONLY on the given details, answer the question: Does the response adhere to the following criterion:.\n"
        "Mentions sunlight.\n\n"
        "Your answer should be a score on a scale of 1 to 5, and should be formatted in XML tags as follows: "
        "<decision>ANSWER</decision>.\n"
        "Provide only the final answer, without additional explanations.\n\n"
        "If the criterion is not applicable to the request and response, answer 5.\n\n"
        "<request>\nWrite the 'Solar' section.\n</request>\n\n"
        f"<response>\n{DOC}\n</response>"
    )


async def test_a_user_message_is_required_even_when_the_request_is_withheld(judge):
    with pytest.raises(EvaluationError):
        await METRIC_REGISTRY["light_absolute_critic"](
            _completion(),
            metric_args={
                "critic_model_name": "judge",
                "criteria": ["Mentions sunlight."],
                "pass_prompt_to_llm_judge": False,
            },
        )

    assert judge.prompts == []


async def test_a_task_with_an_errored_check_runs_every_check_then_has_no_score(judge):
    judge.fail = True
    evaluations = [
        {
            **ArtifactExistanceConstraint(criterion="", should_succeed=True).to_evaluation_dict(),
            **SOLAR,
            "weight": 0.5,
            "evaluation_id": "structure_0",
        },
        {
            **ArtifactLLMJudgeConstraint(
                criterion="x", critic_model_name="judge", custom_user_prompt="Write it."
            ).to_evaluation_dict(),
            **SOLAR,
            "weight": 0.5,
            "evaluation_id": "content_1",
        },
    ]

    with pytest.raises(EvaluationError, match="1 of 2 evaluations failed") as error:
        await constraint_reward(_completion(), {"evaluations": evaluations})

    assert error.value.agg_eval_scores == [1.0, None]
    assert error.value.agg_eval_ids == ["structure_0", "content_1"]


async def test_judge_model_overrides_the_judge_of_every_judge_check_and_nothing_else(judge):
    judge_check = {
        "metric_name": "light_absolute_critic",
        "metric_args": {"critic_model_name": "judge", "criteria": ["Mentions sunlight."], "messages": REQUEST},
        **SOLAR,
        "weight": 0.5,
        "evaluation_id": "content_0",
    }
    verifiable_check = {
        "metric_name": "lexical_match",
        "metric_args": {"string": "sunlight", "match_type": "contains", "should_match": True, "fuzzy_match": False},
        **SOLAR,
        "weight": 0.5,
        "evaluation_id": "content_1",
    }
    info = {"evaluations": [judge_check, verifiable_check]}

    result = await constraint_reward(_completion(), info, judge_model="other-judge")

    assert [model for model, _ in judge.prompts] == ["other-judge"]
    assert result["agg_eval_scores"] == [1.0, 1.0]
    assert info["evaluations"][0]["metric_args"]["critic_model_name"] == "judge"  # the task is left as loaded


async def test_without_judge_model_each_check_is_judged_by_the_model_it_names(judge):
    evaluations = [
        {
            "metric_name": "light_absolute_critic",
            "metric_args": {"critic_model_name": name, "criteria": ["Mentions sunlight."], "messages": REQUEST},
            **SOLAR,
            "weight": 0.5,
            "evaluation_id": f"content_{i}",
        }
        for i, name in enumerate(["judge-a", "judge-b"])
    ]

    await constraint_reward(_completion(), {"evaluations": evaluations})

    assert [model for model, _ in judge.prompts] == ["judge-a", "judge-b"]
