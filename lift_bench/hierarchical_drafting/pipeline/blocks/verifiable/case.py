from dataclasses import dataclass
from typing import Any
import random


# This is required so the verbalizer is registered

from lift_bench.hierarchical_drafting.constraints.case import (
    CaseType,
    VerifiableCaseTypeConstraint,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode

rng = random.Random(1337)


def get_random_case_type():
    return rng.choice(list(CaseType))


@dataclass(kw_only=True)
class CaseTypeConstraintsBlock(MapperBlock):
    version: int = 11

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        node = DocumentNode.from_md(item["document_outline"])

        all_children = children = node.get_children(recursive=True)

        constraints_dct = {}
        for child in children:
            header_text = child.value
            case_type_constraint = VerifiableCaseTypeConstraint(case_type=get_random_case_type())
            case_type_constraint_metadata = VerifiableConstraintMetadata(
                constraint=case_type_constraint, artifact_html_tag=None, check_title=False
            )
            constraints_dct[header_text] = [case_type_constraint_metadata.to_json()]
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "case_type"
        item["constraint_type"] = "verifiable_constraints"

        self.logger.info(
            "Added case-type constraints to {n_sampled_headers} out of {n_total_headers} headers".format(
                n_sampled_headers=len(children),
                n_total_headers=len(all_children),
            )
        )

        return item
