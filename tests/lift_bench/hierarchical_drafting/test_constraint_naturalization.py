import json
from types import SimpleNamespace

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
from lift_bench.hierarchical_drafting.pipeline.blocks import (
    constraint_naturalization,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.constraint_naturalization import (
    ConstraintNaturalizationBlock,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization import (
    get_task_verbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_constraint_text,
)

CASE = VerifiableCaseTypeConstraint(case_type=CaseType.LOWER)
EMOJI = VerifiableEmojiCountConstraint(lower_bound=1, upper_bound=3)


def _metadata(constraint, custom_text: str | None = None) -> VerifiableConstraintMetadata:
    return VerifiableConstraintMetadata(
        constraint=constraint, artifact_html_tag=None, check_title=False, custom_text=custom_text
    )


def _custom_texts(item: dict, col: str = "verifiable_constraints") -> list[str | None]:
    return [
        VerifiableConstraintMetadata.from_json(c).custom_text
        for header in item[col].values()
        for constraints in header.values()
        for c in constraints
    ]


def _fake_completions(*responses: str):
    calls: list[list] = []

    async def fake(prompts, **_kwargs):
        calls.append(prompts)
        return [SimpleNamespace(completions=[responses[i]]) for i in range(len(prompts))]

    return fake, calls


def test_custom_text_overrides_the_templated_text():
    assert render_constraint_text(_metadata(CASE, custom_text="Keep it all lowercase.")) == "Keep it all lowercase."


def test_without_custom_text_the_template_is_used():
    assert render_constraint_text(_metadata(CASE)) == "All the letters in the text should be lowercase"


def test_metadata_serialized_before_custom_text_existed_still_loads():
    payload = json.loads(_metadata(CASE).to_json())
    del payload["custom_text"]

    assert VerifiableConstraintMetadata.from_json(json.dumps(payload)).custom_text is None


@pytest.mark.parametrize("verbalizer_name", ["trailing_constraints", "inline_constraints", "flat", "ifeval_flat"])
async def test_every_verbalizer_uses_the_naturalized_text(verbalizer_name):
    constraints = {
        "Solar": {"case_type": [_metadata(CASE, custom_text="Please keep everything in lowercase").to_json()]}
    }

    messages = await get_task_verbalizer(verbalizer_name).build_messages(
        topic="Energy",
        document_outline="# Energy\n## Solar",
        header_to_constraints_dct=constraints,
        use_strict_header_names=False,
    )

    prompt = messages[0]["content"]
    assert "keep everything in lowercase" in prompt
    assert "All the letters in the text should be lowercase" not in prompt


async def test_naturalization_writes_custom_text_in_batches(tmp_path, monkeypatch):
    fake, calls = _fake_completions(
        "```\n- Keep it lowercase.\n- Use one to three emojis.\n```", "```\n- Add an emoji.\n```"
    )
    monkeypatch.setattr(constraint_naturalization, "generate_completions", fake)
    block = ConstraintNaturalizationBlock(
        inputs_path=str(tmp_path / "in.jsonl"), output_dir=str(tmp_path), batch_size=2
    )
    item = {
        "verifiable_constraints": {
            "Solar": {"case_type": [_metadata(CASE).to_json()], "emoji_count": [_metadata(EMOJI).to_json()]},
            "Wind": {"emoji_count": [_metadata(EMOJI).to_json()]},
        }
    }

    item = await block.process_item(item, {})

    assert len(calls[0]) == 2
    assert _custom_texts(item) == ["Keep it lowercase.", "Use one to three emojis.", "Add an emoji."]


async def test_a_malformed_batch_keeps_the_templated_text(tmp_path, monkeypatch):
    fake, _ = _fake_completions("```\n- Only one item for two constraints.\n```")
    monkeypatch.setattr(constraint_naturalization, "generate_completions", fake)
    block = ConstraintNaturalizationBlock(inputs_path=str(tmp_path / "in.jsonl"), output_dir=str(tmp_path))
    item = {"verifiable_constraints": {"Solar": {"case_type": [_metadata(CASE).to_json(), _metadata(EMOJI).to_json()]}}}

    item = await block.process_item(item, {})

    assert _custom_texts(item) == [None, None]


async def test_structure_constraints_are_not_naturalized(tmp_path, monkeypatch):
    fake, calls = _fake_completions()
    monkeypatch.setattr(constraint_naturalization, "generate_completions", fake)
    block = ConstraintNaturalizationBlock(inputs_path=str(tmp_path / "in.jsonl"), output_dir=str(tmp_path))
    item = {"verifiable_constraints": {"Solar": {"doc_structure": [_metadata(CASE).to_json()]}}}

    item = await block.process_item(item, {})

    assert calls == []
    assert _custom_texts(item) == [None]


@pytest.mark.parametrize(
    "response",
    ["no code block", "```\n- [unclosed\n```", "```\n- one\n- two\n- three\n```", "```\n- one\n- ''\n```"],
)
def test_unusable_responses_are_rejected(response):
    assert ConstraintNaturalizationBlock._parse_response(response, expected_count=2) is None
