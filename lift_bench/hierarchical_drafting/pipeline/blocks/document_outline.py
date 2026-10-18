import re
from dataclasses import dataclass
from typing import Any

from lift_bench.hierarchical_drafting.core.base_blocks import (
    BatchMapperBlock,
)
from lift_bench.hierarchical_drafting.document.document_node import DocumentNode
from lift_bench.hierarchical_drafting.utils.inference import (
    generate_completions,
)
from lift_bench.hierarchical_drafting.document.document_node_verbalizers import (
    HeadersDocumentVerbalizer,
    SimpleDocumentVerbalizer,
)

# TODO: include tone, etc. (maybe as constraints?)
DOCUMENT_OUTLINE_FROM_TEMPLATE_PROMPT_FORMAT = """
I'm writing a document about `{topic}`. Suggest an outline to the document with the exact following skeleton:
```
{document_template}
```

Provide an outline with that exact skeleton, don't provide any additional headers.
Return the outline inside a markdown code block, like the skeleton above.
""".strip()


@dataclass(kw_only=True)
class DocumentOutlineBlock(BatchMapperBlock):
    version: int = 2
    model_name: str
    max_tokens: int
    temperature: float
    template_verbalizer_name: str

    async def process_items(self, items: list[dict], prep_data: dict[str, Any]) -> list[dict]:
        if self.template_verbalizer_name == "headers":
            verbalizer = HeadersDocumentVerbalizer()
        elif self.template_verbalizer_name == "simple":
            verbalizer = SimpleDocumentVerbalizer()
        else:
            raise Exception(f"Unknown template verbalizer name '{self.template_verbalizer_name}'")

        valid_items, invalid_items = [], []

        for item in items:
            if item.get("error", None) is not None:
                invalid_items.append(item)
            else:
                valid_items.append(item)

        messages_lst = []

        for item in valid_items:
            # generate outline from document skeleton
            doc_template = DocumentNode.from_md(item["document_template"], only_skeleton=True)
            messages = [
                {
                    "role": "user",
                    "content": DOCUMENT_OUTLINE_FROM_TEMPLATE_PROMPT_FORMAT.format(
                        topic=item["topic"],
                        document_template=verbalizer.verbalize(doc_template),
                    ),
                },
            ]
            messages_lst.append(messages)

        responses = await generate_completions(
            model=self.model_name,
            prompts=messages_lst,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # check if response contains a valid outline that matches the provided skeleton structure.
        for item, response in zip(valid_items, responses, strict=True):
            doc_template = DocumentNode.from_md(item["document_template"], only_skeleton=True)
            text_response = response.completions[0]
            matches = re.findall(r"```[^\n]*\n(.*?)```", text_response, re.DOTALL)
            if len(matches) == 0:
                item["error"] = f"No MD block found in response: {text_response}"
                continue

            doc_outline = DocumentNode.from_md(matches[0])
            if len(doc_outline.children) == 0:
                item["error"] = f"MD block doesn't contain headers template in response: {matches[0]}"
                continue

            if doc_outline.compare(doc_template):
                item["document_outline"] = doc_outline.to_md()
                item["llm_judge_constraints"] = {}
            else:
                item["error"] = (
                    "Generated outline does not match expected skeleton. \n"
                    "Document skeleton:\n"
                    f"{doc_template.to_md()}\n"
                    "Generated outline:\n"
                    f"{doc_outline.to_md()}"
                )
        return valid_items + invalid_items
