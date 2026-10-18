from dataclasses import dataclass, field
from typing import Any, Literal

from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraint,
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode
from lift_bench.hierarchical_drafting.constraints.artifact import (
    ArtifactLLMJudgeConstraint,
)
import random

rng = random.Random(1337)

TONE_PROMPT_FORMAT = "The text should be written with {tone} tone."


@dataclass(kw_only=True)
class ToneConstraintsBlock(MapperBlock):
    llm_judge_critic_params: dict = field(default_factory=dict)
    eval_type: Literal["light_critic"]
    # TODO: add field `p` - percentage of examples that will contain tone constraints for sampling.

    @property
    def tone_choices(self) -> list[str]:
        return [
            "high-school level",
            "kindergarten level",
            "humoristic",
            "sarcastic",
            "professional",
            # 'elaborative',
            # 'concise',
        ]

    def get_constraint(self, topic: str, header_name: str, criterion: str) -> VerifiableConstraint:
        custom_user_prompt = f"I'm writing a document about '{topic}'. Write content for the section '{header_name}'."
        return ArtifactLLMJudgeConstraint(
            custom_user_prompt=custom_user_prompt,
            criterion=criterion,
            **self.llm_judge_critic_params,
        )

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        document_outline = item["document_outline"]
        topic = item["topic"]
        node = DocumentNode.from_md(document_outline)
        leaf_nodes = [child for child in node.get_children(recursive=True) if child.is_leaf()]

        n_covered = 0
        constraints_dct = {}
        for node in leaf_nodes:
            header_text = node.value
            n_covered += 1

            sampled_tone_type = rng.choice(self.tone_choices)
            criterion = TONE_PROMPT_FORMAT.format(tone=sampled_tone_type)
            constraint = self.get_constraint(topic, header_text, criterion)
            constraint_metadata = VerifiableConstraintMetadata(
                constraint=constraint,
                artifact_html_tag=None,
                check_title=False,
            )
            constraints_dct[header_text] = [constraint_metadata.to_json()]
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "tone"
        item["constraint_type"] = "llm_judge_constraints"

        self.logger.info(
            "Got tone constraints for {n_covered} out of {n_total} topics".format(
                n_total=len(leaf_nodes),
                n_covered=n_covered,
            )
        )

        return item
