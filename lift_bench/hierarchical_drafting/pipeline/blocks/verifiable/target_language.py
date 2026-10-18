from dataclasses import dataclass
from typing import Any
import random


# This is required so the verbalizer is registered
from lift_bench.hierarchical_drafting.constraints.target_language import (
    VerifiableTargetLanguageConstraint,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode

rng = random.Random(1337)

DEFAULT_LANGUAGE_LIST = [
    "en",
    "fr",
    "es",
    "pt",
    "de",
    "ru",
    "it",
    "nl",
    "ar",
    "he",
]


def get_random_language_code():
    return rng.choice(DEFAULT_LANGUAGE_LIST)


@dataclass(kw_only=True)
class TargetLanguageConstraintsBlock(MapperBlock):
    version: int = 11

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        node = DocumentNode.from_md(item["document_outline"])

        all_children = children = node.get_children(recursive=True)

        constraints_dct = {}
        for child in children:
            header_text = child.value
            language_constraint = VerifiableTargetLanguageConstraint(language_code=get_random_language_code())
            language_constraint_metadata = VerifiableConstraintMetadata(
                constraint=language_constraint, artifact_html_tag=None, check_title=False
            )
            constraints_dct[header_text] = [language_constraint_metadata.to_json()]
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "language"
        item["constraint_type"] = "verifiable_constraints"

        self.logger.info(
            "Added language constraints to {n_sampled_headers} out of {n_total_headers} headers".format(
                n_sampled_headers=len(children),
                n_total_headers=len(all_children),
            )
        )

        return item
