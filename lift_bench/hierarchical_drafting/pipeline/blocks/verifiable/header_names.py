from dataclasses import dataclass
from typing import Any
import random


# This is required so the verbalizer is registered

from lift_bench.hierarchical_drafting.constraints.lexical import (
    MatchType,
    VerifiableLexicalConstraint,
)
from lift_bench.hierarchical_drafting.constraints.artifact import (
    ArtifactExistanceConstraint,
)
from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode


@dataclass(kw_only=True)
class HeaderNameConstraintsBlock(MapperBlock):
    version: int = 11
    max_sampled_headers: int = 0

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        node = DocumentNode.from_md(item["document_outline"])

        children = node.get_children(recursive=True)

        if self.max_sampled_headers > 0:
            sample_size = min(len(children), self.max_sampled_headers)
            children = random.sample(children, sample_size)

        for child in children:
            header_text = child.value
            name_constraint = VerifiableLexicalConstraint(
                string=header_text,
                match_type=MatchType.EXACT,
                should_match=True,
                fuzzy_match=False,
            )
            name_constraint_metadata = VerifiableConstraintMetadata(
                constraint=name_constraint, artifact_html_tag=None, check_title=True
            )
            header_item = item["sampled_verifiable_constraints"].get(header_text, {})
            header_item["doc_structure"] = [name_constraint_metadata.to_json()]

            # only for leafs, check that they contain at least one paragraph
            if child.is_leaf():
                paragraph_constraint_metadata = VerifiableConstraintMetadata(
                    constraint=ArtifactExistanceConstraint(criterion="", should_succeed=True),
                    check_title=False,
                    artifact_html_tag=HTMLTag.P,
                    artifact_html_tag_idx=0,
                )
                header_item["doc_structure"].append(paragraph_constraint_metadata.to_json())

            item["sampled_verifiable_constraints"][header_text] = header_item

        return item
