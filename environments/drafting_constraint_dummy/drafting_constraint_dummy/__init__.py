"""Dummy taskset for smoke-testing constraint_reward via vf-eval.

One hardcoded task with three verifiable constraints, so the model response can be scored
deterministically (reward in [0, 1]). Scored like `drafting_constraint`, whose task it reuses.

Run with:
    ./run_eval.sh
"""

from __future__ import annotations

import verifiers.v1 as vf
from drafting_constraint import DraftingData, DraftingTask

_SYSTEM_PROMPT = "You are a helpful assistant. Follow every instruction in the user message precisely."

_USER_PROMPT = (
    "Write exactly one sentence that:\n"
    "1. Contains between 1 and 3 emojis.\n"
    "2. Includes the word 'hello'.\n"
    "3. Is written entirely in lowercase.\n"
    "Do not add any explanation, just the sentence."
)

_ANSWER = {
    "evaluations": [
        {
            "metric_name": "emoji_count",
            "weight": 0.4,
            "metric_args": {"lower_bound": 1, "upper_bound": 3},
        },
        {
            "metric_name": "lexical_match",
            "weight": 0.4,
            "metric_args": {"string": "hello", "match_type": "contains", "should_match": True, "fuzzy_match": True},
        },
        {
            "metric_name": "case_type",
            "weight": 0.2,
            "metric_args": {"case_type": "lower"},
        },
    ]
}


class DraftingConstraintDummyTaskset(vf.Taskset[DraftingTask, vf.TasksetConfig]):
    def load(self) -> list[DraftingTask]:
        data = DraftingData(
            id="0",
            prompt=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT},
            ],
            evaluations=_ANSWER["evaluations"],
        )
        return [DraftingTask(data, self.config.task)]


__all__ = ["DraftingConstraintDummyTaskset"]
