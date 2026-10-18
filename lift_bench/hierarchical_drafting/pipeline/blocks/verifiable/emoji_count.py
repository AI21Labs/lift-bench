from dataclasses import dataclass
from typing import Any
import random


# This is required so the verbalizer is registered
from lift_bench.hierarchical_drafting.constraints.emoji import (
    VerifiableEmojiCountConstraint,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode


rng = random.Random(1337)


@dataclass(kw_only=True)
class EmojiCountConstraintsBlock(MapperBlock):
    version: int = 11
    prob_closed_range: float = 0.6

    def sample_lexical_count_range(self):
        prob_open_ended = (1.0 - self.prob_closed_range) / 2
        range_type = rng.choices(
            population=["open_left", "open_right", "closed"],
            weights=[prob_open_ended, prob_open_ended, self.prob_closed_range],
        )[0]
        left_boundry = right_boundry = None
        match range_type:
            case "open_left":
                right_boundry = rng.choice(range(1, 7))
            case "open_right":
                left_boundry = rng.choice(range(1, 4))
            case "closed":
                left_boundry = rng.choice(range(1, 4))
                right_boundry = left_boundry + rng.choice(range(2, 4))
            case _:
                raise Exception("Unknown range_type")
        return left_boundry, right_boundry

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        node = DocumentNode.from_md(item["document_outline"])

        all_children = children = node.get_children(recursive=True)

        constraints_dct = {}
        for child in children:
            header_text = child.value

            lexical_range_min, lexical_range_max = self.sample_lexical_count_range()
            lexical_count_constraint = VerifiableEmojiCountConstraint(
                lower_bound=lexical_range_min,
                upper_bound=lexical_range_max,
            )
            language_constraint_metadata = VerifiableConstraintMetadata(
                constraint=lexical_count_constraint, artifact_html_tag=None, check_title=False
            )
            constraints_dct[header_text] = [language_constraint_metadata.to_json()]
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "emoji_count"
        item["constraint_type"] = "verifiable_constraints"

        self.logger.info(
            "Added emoji count constraints to {n_sampled_headers} out of {n_total_headers} headers".format(
                n_sampled_headers=len(children),
                n_total_headers=len(all_children),
            )
        )

        return item
