from dataclasses import dataclass, field
from typing import Any
import random


from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode
from lift_bench.hierarchical_drafting.pipeline.utils import (
    sample_dict_across_sublists,
)

DRAFTING_TASK_PROMPT_TEMPLATE = """
Write a markdown document about '{topic}' that has the following outline:
{document_outline}

In addition, follow the constraints below:
{verbalized_constraints}
""".strip()
rng = random.Random(1337)


@dataclass(kw_only=True)
class ConstraintSamplingBlock(MapperBlock):
    version: int = 11
    max_sampled_constraints: int = 0
    sample_per_header: bool = True
    max_sampled_headers: int = 0
    constraints_type_col: str
    exact_sampling: bool = True
    verbose: bool = field(default=False, metadata={"relevant": False})

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        # don't sample any constraint
        if self.max_sampled_constraints == 0:
            item[f"sampled_{self.constraints_type_col}"] = {}
            return item

        # use all constraints
        if self.max_sampled_constraints == -1:
            item[f"sampled_{self.constraints_type_col}"] = item[self.constraints_type_col]
            return item

        # parse document and leaf sections
        node = DocumentNode.from_md(item["document_outline"])
        children = node.get_children(recursive=True)

        # TODO: currently, filter only leaves, might want to apply constraints for non-leaf items.
        children = [child for child in children if child.is_leaf()]

        if self.max_sampled_headers > 0:
            sample_size = min(len(children), self.max_sampled_headers)
            children = rng.sample(children, sample_size)

        # filter `item["llm_judge_constraints"]` to include entries only of `children`
        children_llm_judge_constraints = {
            header_name: constraints_lst
            for header_name, constraints_lst in item[self.constraints_type_col].items()
            if header_name in [child.value for child in children]
        }

        # sample constraints
        if self.sample_per_header:
            item[f"sampled_{self.constraints_type_col}"] = {
                header_name: sample_dict_across_sublists(
                    header_content_constraints,
                    self.max_sampled_constraints,
                    exact=self.exact_sampling,
                    random_state=rng,
                )
                for header_name, header_content_constraints in children_llm_judge_constraints.items()
            }
        else:
            item[f"sampled_{self.constraints_type_col}"] = sample_dict_across_sublists(
                children_llm_judge_constraints,
                self.max_sampled_constraints,
                exact=self.exact_sampling,
                random_state=rng,
            )

        if self.verbose:
            self.logger.info(
                "Sampled {n_sampled_constraints} of {n_sampled_headers} headers out of {n_total_constraints} of {n_total_headers} headers".format(
                    n_sampled_constraints=sum(len(v) for v in item[f"sampled_{self.constraints_type_col}"].values()),
                    n_sampled_headers=len(item[f"sampled_{self.constraints_type_col}"].keys()),
                    n_total_constraints=sum(len(v) for v in item[self.constraints_type_col].values()),
                    n_total_headers=len(item[self.constraints_type_col].keys()),
                )
            )

        return item
