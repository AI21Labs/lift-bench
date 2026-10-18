import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from lift_bench.hierarchical_drafting.document.types import DocumentTextType


from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode
from lift_bench.hierarchical_drafting.pipeline.utils import (
    merge_dicts_of_lists,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization import (
    get_task_verbalizer,
    sample_task_verbalizer,
)


def assign_weights(evaluations: list[dict], content_weight: float) -> None:
    """Pre-compute per-evaluation weights: content_weight * mean(content) + (1 - content_weight) * mean(structure).

    When there are no structure evaluations (e.g. flat tasks), the full weight goes to content.
    """
    structure_evals = [e for e in evaluations if e["weight_group"] == "structure"]
    content_evals = [e for e in evaluations if e["weight_group"] == "content"]
    struct_w = (1.0 - content_weight) if structure_evals else 0.0
    content_w = 1.0 - struct_w
    for e in structure_evals:
        e["weight"] = struct_w / len(structure_evals)
    for e in content_evals:
        e["weight"] = content_w / len(content_evals)


@dataclass(kw_only=True)
class DraftingTaskBlock(MapperBlock):
    version: int = 14
    use_strict_header_names: bool
    task_verbalizer_name: Optional[str] = "random"  # "random" → sample randomly from registered verbalizers
    task_mode: Literal["structured", "flat"] = "structured"
    content_weight: float = field(default=0.9, metadata={"relevant": False})

    def _evaluators_for_task(self, document: DocumentNode, header_to_constraints_dct: dict) -> list[dict]:
        """
        Create a list of evaluation tasks, consisting of the constraints sampled for each header.
        The process iterates over the headers, instantiate constraints and convert them to evaluation format and
        add output parsers so the evaluation will be specific to the relevant header only.
        """
        evaluations = []
        # iterate over headers
        for header_name, header_constraints_dct in header_to_constraints_dct.items():
            if self.task_mode == "structured":
                header_node = document.find_node_by_value(header_name)
                header_html_selector = header_node.get_selector()
            else:
                header_html_selector = []

            # iterate over constraint types
            for constraint_type, constraint_metadata_json_lst in header_constraints_dct.items():
                assert " " not in constraint_type, f"Invalid constraint type {constraint_type} (replace spaces with _)"
                html_tag_idx_iter = itertools.count()  # infinite counter starting at zero
                for constraint_metadata_json in constraint_metadata_json_lst:
                    constraint_metadata = VerifiableConstraintMetadata.from_json(constraint_metadata_json)
                    current_eval_dct = constraint_metadata.constraint.to_evaluation_dict()
                    current_html_selector = header_html_selector
                    # TODO: make sure somehow there are no mixed configurations on same artifact (one with custom index and some without)
                    #  If they overlap (i.e. same index), they will be verbalized separately but will be eval'd on the same artifact, it's confusing for the LLM.
                    #  The only exception is overlap with policy - checking that (1) artifact exists and (2) artifact contains ... - that's ok.
                    # if constraint refer to an HTML tag (e.g., codeblock, table or list), make sure only the i-th instance is parsed
                    if constraint_metadata.artifact_html_tag is not None:
                        current_html_selector = current_html_selector + [
                            constraint_metadata.artifact_html_tag,
                            constraint_metadata.artifact_html_tag_idx or next(html_tag_idx_iter),
                        ]  # not using `+=`, it changes the original list!
                    selector_lst = list(zip(current_html_selector[::2], current_html_selector[1::2]))
                    if selector_lst:
                        current_eval_dct["parser_name"] = "nested_html"
                        current_eval_dct["parser_args"] = {
                            "selector_lst": selector_lst,
                            "is_shallow": constraint_metadata.check_title,
                            "document_type": DocumentTextType.MARKDOWN,
                        }
                    eval_id = len(evaluations)

                    # TODO: need to refactor `constraint_type` to be prefixed accordingly (with `policy_`, `content_`, ...)
                    if constraint_type == "doc_structure":
                        current_eval_dct["weight_group"] = "structure"
                        current_eval_dct["evaluation_id"] = f"structure_{constraint_type}_{eval_id}"
                    else:
                        current_eval_dct["weight_group"] = "content"
                        current_eval_dct["evaluation_id"] = f"content_{constraint_type}_{eval_id}"
                    evaluations.append(current_eval_dct)
        assign_weights(evaluations, self.content_weight)
        return evaluations

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        rng = random.Random(hashlib.md5(item["topic"].encode()).hexdigest())
        document = DocumentNode.from_md(item["document_outline"])
        sampled_constraints_dct_lst = [item[col] for col in item.keys() if col.startswith("sampled_")]
        header_to_constraints_dct = merge_dicts_of_lists(*sampled_constraints_dct_lst)
        evaluations = self._evaluators_for_task(document, header_to_constraints_dct)
        if not any(e["weight_group"] == "content" for e in evaluations):
            item["error"] = "No content constraints were sampled, so the task would be scored on structure alone"
            return item

        verbalizer = (
            sample_task_verbalizer()
            if self.task_verbalizer_name == "random"
            else get_task_verbalizer(self.task_verbalizer_name)
        )
        variations = item.get("prompt_variations") or []
        verbalized_topic = rng.choice(variations) if variations else None
        item["prompt"] = await verbalizer.build_messages(
            topic=item["topic"],
            document_outline=item["document_outline"],
            header_to_constraints_dct=header_to_constraints_dct,
            use_strict_header_names=self.use_strict_header_names,
            verbalized_topic=verbalized_topic,
        )
        item["info"] = {"evaluations": evaluations}
        item["metadata"] = item.get("metadata", {})

        return item
