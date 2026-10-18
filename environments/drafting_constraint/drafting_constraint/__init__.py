"""Verifiers taskset that scores drafting tasks with constraint_reward.

Loads tasks written by the synthesis pipeline (`train.jsonl` / `test.jsonl` from TrainTestSplitBlock):
each row's `prompt` is sent to the model and its `info["evaluations"]` are scored by constraint_reward.
Each trace's `info` keeps the per-evaluation scores and ids (`agg_eval_scores`, `agg_eval_ids`).

Run with (see run_eval.sh for the full set of flags):
    vf-eval drafting_constraint \\
        --env.taskset.dataset-path <output_dir>/<pipeline_name>/TrainTestSplitBlock_<hash>/test.jsonl \\
        --env.agent.harness.id null --env.agent.runtime.type subprocess --no-push \\
        -m <model-name> --client.base-url <endpoint> --client.api-key-var <API_KEY_VAR>
A system prompt for every task is read from a file with `--env.taskset.system-prompt <path>`, and
`--env.taskset.judge-model <model-name>` scores every LLM-judge check with that model instead of the one the
split names.
"""

from __future__ import annotations

from typing import Any

import verifiers.v1 as vf

import lift_bench.hierarchical_drafting.metrics  # noqa: F401  (registers every metric)
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.registry import constraint_reward
from lift_bench.hierarchical_drafting.utils.file_utils import iter_json_lines


class DraftingData(vf.TaskData):
    evaluations: list[dict[str, Any]]
    judge_model: str | None = None
    """Scores every LLM-judge check in `evaluations` instead of the model each names."""


class DraftingTask(vf.Task[DraftingData]):
    @property
    def key(self) -> str:
        return self.data.id or self.hash

    @vf.reward
    async def constraint_reward(self, trace: vf.Trace) -> float:
        """constraint_reward's score. A task with a check that can't run has no score: the error is raised, so
        verifiers records it on the trace and leaves the reward unscored (0.0)."""
        replies = trace.assistant_messages
        # Not trace.last_reply, which strips the answer; checks read it as sent.
        completion = [{"role": "assistant", "content": (replies[-1].content if replies else None) or ""}]
        try:
            result = await constraint_reward(
                completion=completion, info=self._info(), prompt=self._prompt(), judge_model=self.data.judge_model
            )
        except EvaluationError as exc:
            trace.info.update(agg_eval_scores=exc.agg_eval_scores, agg_eval_ids=exc.agg_eval_ids)
            raise
        trace.info.update(agg_eval_scores=result["agg_eval_scores"], agg_eval_ids=result["agg_eval_ids"])
        return result["score"]

    def _info(self) -> dict[str, Any]:
        return {"evaluations": self.data.evaluations}

    def _prompt(self) -> list[dict[str, Any]]:
        prompt = self.data.prompt
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [m.model_dump(exclude_none=True) for m in prompt or []]
        )
        if self.data.system_prompt is not None:
            messages = [{"role": "system", "content": self.data.system_prompt}, *messages]
        return messages


class DraftingConstraintConfig(vf.TasksetConfig):
    # Optional only because vf-eval builds the config from the taskset id alone before applying CLI flags.
    dataset_path: str | None = None
    """A pipeline split JSONL (`train.jsonl` / `test.jsonl` from TrainTestSplitBlock)."""
    judge_model: str | None = None
    """Model that scores every LLM-judge check, instead of the one each task names."""


class DraftingConstraintTaskset(vf.Taskset[DraftingTask, DraftingConstraintConfig]):
    def load(self) -> list[DraftingTask]:
        if self.config.dataset_path is None:
            raise ValueError("drafting_constraint needs a split to evaluate: set --env.taskset.dataset-path")
        return [
            DraftingTask(
                DraftingData(
                    id=str(row["example_id"]),
                    prompt=row["prompt"],
                    evaluations=row["info"]["evaluations"],
                    judge_model=self.config.judge_model,
                ),
                self.config.task,
            )
            for row in iter_json_lines(self.config.dataset_path)
        ]


__all__ = ["DraftingConstraintTaskset"]
