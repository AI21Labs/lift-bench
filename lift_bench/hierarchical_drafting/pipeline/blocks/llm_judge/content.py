import re
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraint,
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode
from lift_bench.hierarchical_drafting.constraints.artifact import (
    ArtifactLLMJudgeConstraint,
    ArtifactExistanceConstraint,
)
from lift_bench.hierarchical_drafting.utils.inference import (
    generate_completions,
)

# TODO: include tone, etc. (maybe as constraints?)
CONTENT_CONSTRAINTS_PROMPT_TEMPLATE = """
I'm writing a document about '{topic}' with the following outline:
```
{document_outline}
```

Currently, I'm writing the content of the section {header_name}. Provide up to {max_artifacts} suggestions for topics covered in this section.
* Each suggestion should be formatted in a descriptive manner and refer to "The text", (for a header named '## Superheroes Iconic Characters', for example): "The text should provide examples for popular characters from the Marvel and DC universes with a short introduction.".
* Each item is independent from the others, so don't start an item by "In addition, this text should ...".


Your output should be YAML-formatted list, inside a markdown code block as follows:
```
- The text should explain the core software components of the iPhone ecosystem and system-level services.
- The text should provide a list detailing wireless and connectivity technologies used by Playstation consoles.
- The text should contain a recipe for making Creme Brulee.
- The text should review the pros and cons of Kanban and Scrum agile methodologies.
etc.
```
""".strip()


@dataclass(kw_only=True)
class ContentConstraintsBlock(MapperBlock):
    max_constraints: int
    model_name: str
    max_tokens: int
    temperature: float
    llm_judge_critic_params: dict = field(default_factory=dict)
    eval_type: Literal["artifact_exists", "light_critic"]

    def get_constraint(self, topic: str, header_name: str, criterion: str) -> VerifiableConstraint:
        if self.eval_type != "artifact_exists":
            custom_user_prompt = (
                f"I'm writing a document about '{topic}'. Write content for the section '{header_name}'."
            )
            return ArtifactLLMJudgeConstraint(
                custom_user_prompt=custom_user_prompt,
                criterion=criterion,
                **self.llm_judge_critic_params,
            )
        else:
            return ArtifactExistanceConstraint(criterion=criterion, should_succeed=True)

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        document_outline = item["document_outline"]
        topic = item["topic"]
        node = DocumentNode.from_md(document_outline)
        leaf_nodes = [child for child in node.get_children(recursive=True) if child.is_leaf()]

        header_text_lst = [node.value for node in leaf_nodes]
        constraint_suggestion_prompt_lst = [
            CONTENT_CONSTRAINTS_PROMPT_TEMPLATE.format(
                topic=topic,
                document_outline=document_outline,
                header_name=header_name,
                max_artifacts=self.max_constraints,
            )
            for header_name in header_text_lst
        ]
        response_lst = await self.get_suggestions(constraint_suggestion_prompt_lst)
        n_covered = 0
        constraints_dct = {}
        for header_text, text_response in zip(header_text_lst, response_lst, strict=True):
            matches = re.findall(r"```[^\n]*\n(.*?)```", text_response, re.DOTALL)
            if len(matches) == 0:
                item["error"] = f"No yaml block found in response: {text_response}"
                return item

            try:
                content_constraints = yaml.safe_load(matches[0])
            except Exception:
                item["error"] = f"Error parsing the yaml block: {matches[0]}"
                return item

            if not (isinstance(content_constraints, list) and len(content_constraints) > 0):
                continue
            n_covered += 1

            constraints = [self.get_constraint(topic, header_text, criterion) for criterion in content_constraints]
            constraints_metadata = [
                VerifiableConstraintMetadata(
                    constraint=constraint,
                    # if not using an LLM judge, check that at least 1 paragraph exists in the section's content
                    artifact_html_tag=HTMLTag.P if self.eval_type == "artifact_exists" else None,
                    check_title=False,
                ).to_json()
                for constraint in constraints
            ]
            constraints_dct[header_text] = constraints_metadata
        item["constraint_dct"] = constraints_dct
        item["constraint_name"] = "content"
        item["constraint_type"] = "llm_judge_constraints"

        self.logger.info(
            "Got content constraints for {n_covered} out of {n_total} topics".format(
                n_total=len(leaf_nodes),
                n_covered=n_covered,
            )
        )

        return item

    async def get_suggestions(self, prompts: list[str]):
        messages_lst = [
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            for prompt in prompts
        ]
        response_lst = await generate_completions(
            model=self.model_name,
            prompts=messages_lst,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text_response_lst = [response.completions[0] for response in response_lst]
        return text_response_lst
