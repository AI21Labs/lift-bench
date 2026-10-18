from dataclasses import dataclass
from typing import Any, Optional
import random

from lift_bench.hierarchical_drafting.iftrain.instructions_registry import (
    INSTRUCTION_DICT,
    INSTRUCTION_CONFLICTS,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.constraints.ifbench import (
    IFBenchConstraint,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode

rng = random.Random(1337)  # set global randomness


def random_list_with_sum(N, M) -> list[int]:
    """
    Generate a list of N non-negative integers that sum to M.

    Parameters:
        N (int): Number of elements in the list
        M (int): The desired total sum of the list

    Returns:
        list: A list of N integers that sum to M
    """
    if N <= 0 or M < 0:
        raise ValueError("N must be > 0 and M must be >= 0")

    # Step 1: Pick N-1 random breakpoints between 0 and M
    cuts = sorted(rng.sample(range(M + N - 1), N - 1))

    # Step 2: Use these breakpoints to determine segment sizes
    result = [cuts[0]]
    for i in range(1, len(cuts)):
        result.append(cuts[i] - cuts[i - 1])
    result.append(M + N - 1 - cuts[-1])

    # Step 3: Subtract 1 from each to get values summing to M
    final_result = [x - 1 for x in result]

    return final_result


class IFBenchInstructionSampler:
    def __init__(self, instruction_id_lst: list[str] = None):
        if instruction_id_lst is not None:
            self.instruction_id_lst: list[str] = instruction_id_lst
        else:
            self.instruction_id_lst: list[str] = list(INSTRUCTION_DICT.keys())

    def sample(self, n) -> list[str]:
        instruction_id_candidates = self.instruction_id_lst.copy()
        rng.shuffle(instruction_id_candidates)

        # sample instructions, make sure there are no conflicts
        sampled_instruction_id_set = set()
        for instruction_id in instruction_id_candidates:
            if len(sampled_instruction_id_set) == n:
                break
            if sampled_instruction_id_set.isdisjoint(INSTRUCTION_CONFLICTS[instruction_id]):
                sampled_instruction_id_set.add(instruction_id)

        return list(sampled_instruction_id_set)


@dataclass(kw_only=True)
class IFBenchConstraintsBlock(MapperBlock):
    version: int = 11
    max_sampled_constraints: int = 3
    custom_instructions_lst: Optional[list[str]] = None

    def __post_init__(self):
        unknown = sorted(set(self.custom_instructions_lst or []) - INSTRUCTION_DICT.keys())
        if unknown:
            raise ValueError(f"Unknown IFTrain instruction ids: {unknown}")

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        # parse document
        node = DocumentNode.from_md(item["document_outline"])
        children = node.get_children(recursive=True)

        # TODO: currently, filter only leaves, might want to apply constraints for non-leaf items.
        children = [child for child in children if child.is_leaf()]

        # get instructions pool and initialize sampler
        if_sampler = IFBenchInstructionSampler(self.custom_instructions_lst)

        # sample instructions for each header
        constraints_dct = {}
        for child in children:
            header_text = child.value
            n_sample = rng.randint(1, self.max_sampled_constraints)
            sampled_instructions = if_sampler.sample(n_sample)
            sampled_constraints = await self._create_constraints_from_instructions(sampled_instructions)
            constraints_metadata = [
                VerifiableConstraintMetadata(constraint=constraint, artifact_html_tag=None, check_title=False).to_json()
                for constraint in sampled_constraints
            ]
            constraints_dct[header_text] = constraints_metadata
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "ifbench"
        item["constraint_type"] = "verifiable_constraints"

        self.logger.info(
            "Sampled {n_sampled_constraints} to {n_sampled_headers} headers out of {n_total_headers} headers".format(
                n_sampled_constraints=sum(len(v) for v in constraints_dct.values()),
                n_sampled_headers=len(constraints_dct.keys()),
                n_total_headers=len(children),
            )
        )

        return item

    async def _create_constraints_from_instructions(self, sampled_instructions: list[str]) -> list[IFBenchConstraint]:
        """
        Instantiate constraints from IFTrain/IFBench instructions.
        """
        results = []
        for instruction_id in sampled_instructions:
            instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
            _ = instruction.build_description()
            instruction_kwargs = instruction.get_instruction_args()
            if_constraint = IFBenchConstraint(instruction_id=instruction_id, instruction_kwargs=instruction_kwargs)
            results.append(if_constraint)
        return results
