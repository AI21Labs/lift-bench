from typing import Optional, Union

from lift_bench.hierarchical_drafting.iftrain.instructions_registry import (
    INSTRUCTION_DICT,
)
from lift_bench.hierarchical_drafting.metrics.registry import (
    register_metric,
)
from lift_bench.hierarchical_drafting.metrics.errors import MetricError


@register_metric("ifbench")
async def ifbench(
    completion,
    instruction_id_list: list[str],
    kwargs: list[dict[str, Optional[Union[list, str, int]]]],
    hard_aggregation: bool = True,
) -> float:
    response = completion[-1]["content"]
    user_messages = [m for m in completion if m["role"] == "user"]
    prompt = user_messages[-1]["content"] if user_messages else ""

    is_following_list = []
    for index, instruction_id in enumerate(instruction_id_list):
        if instruction_id not in INSTRUCTION_DICT:
            raise MetricError(f"unknown instruction id {instruction_id!r}")

        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        clean_kwargs = {k: v for k, v in (kwargs[index] if index < len(kwargs) else {}).items() if v is not None}
        instruction.build_description(**clean_kwargs)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        is_following_list.append(bool(response.strip() and instruction.check_following(response)))

    if hard_aggregation:  # all([]) is True: with no instructions there is nothing to fail
        return 1.0 if all(is_following_list) else 0.0
    if not is_following_list:
        raise MetricError("no instructions to average over")
    return sum(is_following_list) / len(is_following_list)
