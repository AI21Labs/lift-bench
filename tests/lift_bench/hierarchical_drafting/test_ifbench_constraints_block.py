import pytest

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.iftrain.instructions_registry import (
    INSTRUCTION_CONFLICTS,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.iftrain.block import (
    IFBenchConstraintsBlock,
)

INSTRUCTIONS = [
    "copy:repeat_phrase",
    "count:lowercase_counting",
    "first_word:first_word_answer",
    "keywords:palindrome",
    "last_word:last_word_sent",
    "paragraphs:paragraphs",
]
OUTLINE = "# Energy\n## Solar\n## Wind\n### Offshore\n### Onshore"
LEAF_HEADERS = {"Solar", "Offshore", "Onshore"}


def _instruction_id(constraint_json: str) -> str:
    constraint = VerifiableConstraintMetadata.from_json(constraint_json).constraint
    return constraint.to_evaluation_dict()["metric_args"]["instruction_id_list"][0]


def _block(tmp_path, **kwargs) -> IFBenchConstraintsBlock:
    return IFBenchConstraintsBlock(inputs_path=str(tmp_path / "in.jsonl"), output_dir=str(tmp_path), **kwargs)


async def _sampled_instruction_ids(block: IFBenchConstraintsBlock) -> dict[str, list[str]]:
    item = await block.process_item({"topic": "Energy", "document_outline": OUTLINE}, {})
    return {header: [_instruction_id(c) for c in constraints] for header, constraints in item["constraint_dct"].items()}


def test_unknown_instruction_ids_are_rejected_up_front(tmp_path):
    with pytest.raises(ValueError, match="no:such_instruction"):
        _block(tmp_path, custom_instructions_lst=["copy:repeat_phrase", "no:such_instruction"])


async def test_each_leaf_header_gets_between_one_and_max_instructions_from_the_custom_list(tmp_path):
    block = _block(tmp_path, custom_instructions_lst=INSTRUCTIONS, max_sampled_constraints=3)

    for _ in range(20):
        sampled = await _sampled_instruction_ids(block)

        assert set(sampled) == LEAF_HEADERS
        for instruction_ids in sampled.values():
            assert 1 <= len(instruction_ids) <= 3
            assert set(instruction_ids) <= set(INSTRUCTIONS)


async def test_sampled_instructions_never_conflict(tmp_path):
    conflicting = ["length_constraints:number_paragraphs", "length_constraints:nth_paragraph_first_word"]
    assert conflicting[1] in INSTRUCTION_CONFLICTS[conflicting[0]]
    block = _block(tmp_path, custom_instructions_lst=conflicting, max_sampled_constraints=2)

    for _ in range(20):
        for instruction_ids in (await _sampled_instruction_ids(block)).values():
            assert len(instruction_ids) == 1


async def test_instruction_kwargs_are_built_without_a_model(tmp_path):
    block = _block(tmp_path, custom_instructions_lst=["keywords:palindrome"])

    item = await block.process_item({"topic": "Energy", "document_outline": OUTLINE}, {})

    assert _instruction_id(item["constraint_dct"]["Solar"][0]) == "keywords:palindrome"
