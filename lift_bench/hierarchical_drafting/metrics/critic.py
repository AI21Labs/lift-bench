import re
from typing import Literal, Optional

from lift_bench.hierarchical_drafting.metrics.registry import (
    register_metric,
)
from lift_bench.hierarchical_drafting.metrics.errors import MetricError
from lift_bench.hierarchical_drafting.utils.inference import (
    RetryPolicy,
    generate_completions,
)


# The judge prompts, chosen by whether criteria are given and where the instruction goes.

PROMPT_ABSOLUTE = """Denote the following request and response, enclosed by xml tags <request>REQUEST</request> and <response>RESPONSE</response> respectively.

Based ONLY on the given details, answer the question "Is the response answers in a helpful manner to the request?".
* "Helpful manner" can be either (1) answering the request, or (2) explaining why the request cannot be fulfilled.

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations.

<request>
{request}
</request>

<response>
{response}
</response>"""

PROMPT_ABSOLUTE_INSTRUCTION_AFTER = """Denote the following request and response, enclosed by xml tags <request>REQUEST</request> and <response>RESPONSE</response> respectively.

<request>
{request}
</request>

<response>
{response}
</response>

Based ONLY on the given details, answer the question "Is the response answers in a helpful manner to the request?".
* "Helpful manner" can be either (1) answering the request, or (2) explaining why the request cannot be fulfilled.

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations."""

PROMPT_ABSOLUTE_WITH_CRITERION = """Denote the following request and response, enclosed by xml tags <request>REQUEST</request> and <response>RESPONSE</response> respectively.

Based ONLY on the given details, answer the question: Does the response adhere to the following criterion:.
{criterion}

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations.

If the criterion is not applicable to the request and response, answer {max_score}.

<request>
{request}
</request>

<response>
{response}
</response>"""

PROMPT_ABSOLUTE_WITH_CRITERION_INSTRUCTION_AFTER = """Denote the following request and response, enclosed by xml tags <request>REQUEST</request> and <response>RESPONSE</response> respectively.

<request>
{request}
</request>

<response>
{response}
</response>

Based ONLY on the given details, answer the question: Does the response adhere to the following criterion:.
{criterion}

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations.

If the criterion is not applicable to the request and response, answer {max_score}."""

PROMPT_ABSOLUTE_WITH_CRITERION_NO_REQUEST = """Denote the following response by a LLM, enclosed by xml tags <response>RESPONSE</response>.

Answer the question: Does the response adhere to the following instruction:.
{criterion}

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations.

<response>
{response}
</response>"""

PROMPT_ABSOLUTE_WITH_CRITERION_NO_REQUEST_INSTRUCTION_AFTER = """Denote the following response by a LLM, enclosed by xml tags <response>RESPONSE</response>.

<response>
{response}
</response>

Answer the question: Does the response adhere to the following instruction:.
{criterion}

Your answer should be {verbalized_score_range}, and should be formatted in XML tags as follows: <decision>ANSWER</decision>.
Provide only the final answer, without additional explanations."""

WITH_CRITERION_PASS_PROMPT_TO_LLM_JUDGE_INST_AFTER_COMPLETION = {
    True: {
        True: PROMPT_ABSOLUTE_WITH_CRITERION_INSTRUCTION_AFTER,
        False: PROMPT_ABSOLUTE_WITH_CRITERION,
    },
    False: {
        True: PROMPT_ABSOLUTE_WITH_CRITERION_NO_REQUEST_INSTRUCTION_AFTER,
        False: PROMPT_ABSOLUTE_WITH_CRITERION_NO_REQUEST,
    },
}

NO_CRITERION_INST_AFTER_COMPLETION = {
    True: PROMPT_ABSOLUTE_INSTRUCTION_AFTER,
    False: PROMPT_ABSOLUTE,
}


@register_metric("light_absolute_critic")
async def light_absolute_critic(
    completion,
    criteria: Optional[list[str]] = None,
    messages: Optional[list[dict]] = None,
    critic_model_name: str = "google/gemma-4-31B-it",
    temperature: float = 0.5,
    max_tokens: int = 8192,
    n: int = 1,
    valid_judgements_ratio_threshold: float = 0.7,
    good_completion_threshold: float = 0.7,
    threshold_key: Literal["avg_score", "min_score"] = "avg_score",
    use_binary_range: bool = True,
    allow_score_out_of_range: bool = True,
    pass_prompt_to_llm_judge: bool = True,
    instruction_after_completion: bool = False,
    llm_judge_protocol: str = "sfr",
    criterion: Optional[str] = None,
) -> float:
    """LLM judge: 1.0 when the judge's score reaches the threshold, else 0.0.

    `messages` defaults to the conversation preceding the response (the task prompt, which constraint_reward puts
    before the completion). No user message, a failed judge request or too few parseable judgements raise
    MetricError.
    """
    del llm_judge_protocol  # a critic config field the light critic accepts but doesn't use
    if criterion is not None and criteria is None:  # `criterion` is accepted for `criteria`
        criteria = [criterion]
    response = completion[-1]["content"]
    conversation = messages if messages is not None else completion[:-1]
    user_messages = [m for m in conversation if m["role"] == "user"]
    if len(user_messages) == 0:
        raise MetricError("light_absolute_critic: message with role='user' not found")
    user_request = user_messages[-1]

    min_score, max_score = score_range = (0, 1) if use_binary_range else (1, 5)
    if use_binary_range:
        verbalized_score_range = f"either {max_score} for True or {min_score} for False"
    else:
        verbalized_score_range = f"a score on a scale of {min_score} to {max_score}"

    if criteria is not None and len(criteria) == 1:
        prompt = WITH_CRITERION_PASS_PROMPT_TO_LLM_JUDGE_INST_AFTER_COMPLETION[pass_prompt_to_llm_judge][
            instruction_after_completion
        ].format(
            request=user_request["content"],
            response=response,
            criterion=criteria[0],
            verbalized_score_range=verbalized_score_range,
            max_score=max_score,
        )
    else:
        prompt = NO_CRITERION_INST_AFTER_COMPLETION[instruction_after_completion].format(
            request=user_request["content"],
            response=response,
            verbalized_score_range=verbalized_score_range,
        )

    try:
        responses = await generate_completions(
            model=critic_model_name,
            prompts=[[{"role": "user", "content": prompt}]],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            retry_policy=RetryPolicy(),
        )
    except Exception as exc:
        raise MetricError(f"light_absolute_critic: error sending request to model {critic_model_name!r}") from exc

    scores = []
    n_invalid = 0
    for c in responses[0].completions:
        try:
            match = re.search(r"<decision>(.+?)</decision>", c)
            score = int(match.groups()[0])
            if score < score_range[0] or score > score_range[1]:
                if allow_score_out_of_range:
                    score = score_range[0] if score < score_range[0] else score_range[1]
                else:
                    raise ValueError(f"Score {score} is out of range {score_range}")
            scores.append((score - min_score) / (max_score - min_score))
        except Exception:
            n_invalid += 1

    valid_judgements_ratio = len(scores) / (len(scores) + n_invalid)
    if valid_judgements_ratio < valid_judgements_ratio_threshold:
        raise MetricError(
            "light_absolute_critic: valid judgements ratio is less than the threshold: "
            f"{valid_judgements_ratio}<{valid_judgements_ratio_threshold}"
        )

    score = sum(scores) / len(scores) if threshold_key == "avg_score" else min(scores)
    return float(score >= good_completion_threshold)
