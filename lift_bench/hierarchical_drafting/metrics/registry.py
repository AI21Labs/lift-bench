from typing import Any, Callable, Optional

from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.with_output_parser import (
    with_output_parser,
)

METRIC_REGISTRY: dict[str, Callable] = {}
LLM_JUDGE_METRIC = "light_absolute_critic"


def register_metric(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        METRIC_REGISTRY[name] = with_output_parser(func)
        return func

    return decorator


def with_judge_model(evaluation: dict[str, Any], judge_model: str) -> dict[str, Any]:
    """`evaluation` scored by `judge_model` if it is an LLM-judge check, else unchanged."""
    if evaluation.get("metric_name") != LLM_JUDGE_METRIC:
        return evaluation
    return {**evaluation, "metric_args": {**evaluation.get("metric_args", {}), "critic_model_name": judge_model}}


async def constraint_reward(
    completion,
    info: dict[str, Any],
    prompt: Optional[list[dict]] = None,
    judge_model: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute weighted constraint reward over all evaluations in `info`.

    Metrics see the task's `prompt` followed by the `completion`, so a check that reads the request (e.g. the critic
    without `messages`) finds it there. `judge_model` replaces the judge named by every LLM-judge check.

    Returns a dict with:
      score          - final aggregated weighted score
      agg_eval_ids   - list of evaluation_id values per evaluation entry (None when
                       no entry carries an evaluation_id key)
      agg_eval_scores - list of raw (unweighted) per-evaluation scores

    Every check runs; if any couldn't be evaluated, the task has no score and EvaluationError is raised.
    """
    evaluations = info.get("evaluations", [])
    if not evaluations:
        return {"score": 1.0, "agg_eval_ids": None, "agg_eval_scores": []}
    if judge_model is not None:
        evaluations = [with_judge_model(e, judge_model) for e in evaluations]

    for e in evaluations:
        if e.get("metric_name") not in METRIC_REGISTRY:
            raise EvaluationError(
                f"Evaluation {e.get('evaluation_id')!r} names unregistered metric {e.get('metric_name')!r}; "
                f"registered: {sorted(METRIC_REGISTRY)}"
            )

    conversation = [*(prompt or []), *completion]
    agg_eval_scores: list[Optional[float]] = []
    agg_eval_ids: list[Any] = []
    errors = []
    total = 0.0
    for e in evaluations:
        agg_eval_ids.append(e.get("evaluation_id"))
        try:
            score = float(await METRIC_REGISTRY[e["metric_name"]](conversation, **e))
        except EvaluationError as exc:
            agg_eval_scores.append(None)
            errors.append(f"{e.get('evaluation_id')!r}: {exc}")
            continue
        agg_eval_scores.append(score)
        total += e["weight"] * score

    if errors:
        raise EvaluationError(
            f"{len(errors)} of {len(evaluations)} evaluations failed: " + "; ".join(errors),
            agg_eval_scores=agg_eval_scores,
            agg_eval_ids=agg_eval_ids,
        )
    return {
        "score": total,
        "agg_eval_ids": agg_eval_ids if any(v is not None for v in agg_eval_ids) else None,
        "agg_eval_scores": agg_eval_scores,
    }
