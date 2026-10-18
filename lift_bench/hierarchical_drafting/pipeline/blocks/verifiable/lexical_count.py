from dataclasses import dataclass
from typing import Any
import random


# This is required so the verbalizer is registered
from lift_bench.hierarchical_drafting.constraints.lexical import (
    MatchType,
    VerifiableLexicalConstraint,
    VerifiableLexicalCountConstraint,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode

from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.iftrain.keyword_utils import (
    get_common_keywords,
)

rng = random.Random(1337)


@dataclass(kw_only=True)
class LexicalCountConstraintsBlock(MapperBlock):
    version: int = 11
    prob_closed_range: float = 0.6
    strict_word_boundaries: bool = True
    fuzzy_match: bool = True
    keyword_generation_model_name: str
    keyword_generation_max_retries: int

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

            try:
                kw_lst = await get_common_keywords(
                    model_name=self.keyword_generation_model_name,
                    max_retries=self.keyword_generation_max_retries,
                    document_topic=item["topic"],
                    hierarchy_section_names=child.ancestors_alias_list,
                    n_words=3,
                )
            except Exception:
                continue
            lexical_constraints = [self.sample_constraint(kw) for kw in kw_lst]
            constraints_dct[header_text] = [c.to_json() for c in lexical_constraints]
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "lexical_count"
        item["constraint_type"] = "verifiable_constraints"

        self.logger.info(
            "Added lexical count constraints to {n_sampled_headers} out of {n_total_headers} headers".format(
                n_sampled_headers=len(children),
                n_total_headers=len(all_children),
            )
        )

        return item

    def sample_constraint(self, kw: str) -> VerifiableConstraintMetadata:
        constraint_type = rng.choices(
            population=["forbidden", "allowed", "lexical_count"],
            weights=[
                0.15,
                0.15,
                0.7,
            ],
            k=1,
        )[0]
        if constraint_type == "lexical_count":
            lexical_range_min, lexical_range_max = self.sample_lexical_count_range()
            lexical_count_constraint = VerifiableLexicalCountConstraint(
                string=kw,
                lower_bound=lexical_range_min,
                upper_bound=lexical_range_max,
                fuzzy_match=self.fuzzy_match,
                strict_word_boundaries=self.strict_word_boundaries,
            )
        else:
            lexical_count_constraint = VerifiableLexicalConstraint(
                string=kw,
                match_type=MatchType.CONTAINS,
                should_match=constraint_type == "allowed",
            )
        return VerifiableConstraintMetadata(
            constraint=lexical_count_constraint, artifact_html_tag=None, check_title=False
        )
