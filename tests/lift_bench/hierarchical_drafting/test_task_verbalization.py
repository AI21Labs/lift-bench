import os
import random
import subprocess
import sys

import pytest

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.constraints.case import (
    CaseType,
    VerifiableCaseTypeConstraint,
)
from lift_bench.hierarchical_drafting.constraints.emoji import (
    VerifiableEmojiCountConstraint,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.drafting_task import (
    DraftingTaskBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.prompt_variations import (
    PromptVariationsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization import (
    get_task_verbalizer,
)

TOPIC = "Renewable Energy"
OUTLINE = "# Renewable Energy\n## Solar Power"


def _metadata_json(constraint) -> str:
    return VerifiableConstraintMetadata(constraint=constraint, artifact_html_tag=None, check_title=False).to_json()


def _constraints() -> dict:
    return {
        "Solar Power": {
            "case_type": [_metadata_json(VerifiableCaseTypeConstraint(case_type=CaseType.LOWER))],
            "emoji_count": [_metadata_json(VerifiableEmojiCountConstraint(lower_bound=1, upper_bound=3))],
        }
    }


async def _prompt(verbalizer_name: str, topic: str = TOPIC, verbalized_topic: str | None = None) -> str:
    messages = await get_task_verbalizer(verbalizer_name).build_messages(
        topic=topic,
        document_outline=OUTLINE,
        header_to_constraints_dct=_constraints(),
        use_strict_header_names=False,
        verbalized_topic=verbalized_topic,
    )
    return messages[0]["content"]


def test_ifeval_flat_prompt_does_not_depend_on_the_interpreter_hash_salt():
    code = (
        f"import asyncio, sys; sys.path.insert(0, {os.path.dirname(__file__)!r}); "
        "from test_task_verbalization import _prompt; print(asyncio.run(_prompt('ifeval_flat')))"
    )
    prompts = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": salt},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for salt in ("1", "2")
    }
    assert len(prompts) == 1


@pytest.mark.parametrize("verbalizer_name", ["trailing_constraints", "inline_constraints", "flat"])
async def test_verbalized_topic_replaces_the_templated_opening(verbalizer_name):
    variation = "I'm preparing a talk and need an overview of renewable energy."

    prompt = await _prompt(verbalizer_name, verbalized_topic=variation)

    assert prompt.startswith(variation)
    assert f"'{TOPIC}'" not in prompt


@pytest.mark.parametrize("verbalizer_name", ["trailing_constraints", "inline_constraints", "flat"])
async def test_without_a_verbalized_topic_the_prompt_names_the_topic(verbalizer_name):
    prompt = await _prompt(verbalizer_name)

    assert f"'{TOPIC}'" in prompt


@pytest.mark.parametrize(
    "variation", ["Tell me about solar power.", "Can you explain solar power?", "Explain solar power"]
)
async def test_flat_opening_is_followed_by_exactly_one_terminal_mark(variation):
    prompt = await _prompt("flat", verbalized_topic=variation)

    first_line = prompt.split("\n")[0]
    assert first_line.endswith(" Follow these constraints:")
    assert first_line.removesuffix(" Follow these constraints:")[-2:] not in ("..", "?.", "!.")
    assert first_line.removesuffix(" Follow these constraints:")[-1] in ".?!"


async def test_ifeval_flat_prompt_is_reproducible_for_the_same_topic():
    assert await _prompt("ifeval_flat") == await _prompt("ifeval_flat")


@pytest.mark.parametrize("topic", ["Renewable Energy", "Roman Empire", "Jazz", "Quantum Computing", "Bees"])
async def test_ifeval_flat_prompt_ends_with_terminal_punctuation(topic):
    prompt = await _prompt("ifeval_flat", topic=topic)

    assert prompt[-1] in ".?!"


@pytest.mark.parametrize("topic", ["Renewable Energy", "Roman Empire", "Jazz", "Quantum Computing", "Bees"])
async def test_ifeval_flat_never_adds_a_period_after_a_question(topic):
    prompt = await _prompt("ifeval_flat", topic=topic, verbalized_topic=f"Could you tell me about {topic}?")

    assert "?." not in prompt


def test_single_artifact_constraint_renders_as_plain_text():
    verbalizer = get_task_verbalizer("ifeval_flat")

    rendered = verbalizer._render_artifact_section("tables", ["Include a table of costs"], random.Random(0))

    assert rendered == "Include a table of costs"


def test_multiple_artifact_constraints_render_as_a_numbered_block():
    verbalizer = get_task_verbalizer("ifeval_flat")

    rendered = verbalizer._render_artifact_section("tables", ["Table A", "Table B"], random.Random(0))

    assert rendered.endswith("2 tables:\n1. Table A\n2. Table B")


async def test_drafting_task_picks_the_same_variation_for_the_same_topic(tmp_path):
    block = DraftingTaskBlock(
        inputs_path=str(tmp_path / "in.jsonl"),
        output_dir=str(tmp_path),
        use_strict_header_names=False,
        task_verbalizer_name="trailing_constraints",
    )
    variations = [f"Variation number {i} about renewable energy." for i in range(10)]

    def item() -> dict:
        return {
            "topic": TOPIC,
            "document_outline": OUTLINE,
            "sampled_constraints": _constraints(),
            "prompt_variations": variations,
        }

    first = (await block.process_item(item(), {}))["prompt"][0]["content"]
    second = (await block.process_item(item(), {}))["prompt"][0]["content"]

    assert first == second
    assert first.split("\n")[0] in variations


async def test_a_task_without_content_constraints_is_an_error(tmp_path):
    block = DraftingTaskBlock(
        inputs_path=str(tmp_path / "in.jsonl"),
        output_dir=str(tmp_path),
        use_strict_header_names=True,
        task_verbalizer_name="trailing_constraints",
    )
    structure_only = {"Solar Power": {"doc_structure": _constraints()["Solar Power"]["case_type"]}}

    item = await block.process_item(
        {"topic": TOPIC, "document_outline": OUTLINE, "sampled_constraints": structure_only}, {}
    )

    assert "No content constraints" in item["error"]
    assert "info" not in item


def test_prompt_variations_are_parsed_from_the_yaml_code_block():
    text = 'Sure:\n```yaml\n- "I need an overview of the Roman Empire."\n- "short"\n- "Tell me about the Roman Empire."\n```'

    assert PromptVariationsBlock._parse_variations(text) == [
        "I need an overview of the Roman Empire.",
        "Tell me about the Roman Empire.",
    ]


@pytest.mark.parametrize("text", ["no code block here", "```\n- [unclosed\n```", "```\njust: a mapping\n```"])
def test_unparseable_prompt_variations_yield_no_variations(text):
    assert PromptVariationsBlock._parse_variations(text) == []
