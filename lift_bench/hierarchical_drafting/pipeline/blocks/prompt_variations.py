import re
from dataclasses import dataclass
from typing import Any

import yaml

from lift_bench.hierarchical_drafting.core.base_blocks import (
    BatchMapperBlock,
)
from lift_bench.hierarchical_drafting.utils.inference import (
    generate_completions,
)

# TODO: make this configurable / move to a registry like TaskVerbalizer
_PROMPT_TEMPLATE = """
You are given prompts that currently follow a repetitive pattern such as:

"Write about {topic}"

Your task is to rewrite each prompt into multiple diverse, natural-sounding user requests while preserving the same underlying intent: asking for content about the given topic.

Requirements:

* Keep the core topic unchanged.
* Produce prompts with varied phrasing and structures.
* Use different motivations and user goals when appropriate (e.g., preparing a presentation, writing an article, studying, personal curiosity, work tasks, creating educational material, etc.).
* Vary request styles: direct instructions, questions, requests, etc.
* Refrain from specifying task-specific instructions.
* Avoid repetitive templates and avoid merely replacing "Write" with synonyms.
* Make prompts sound like realistic requests from different users.
* Preserve semantic equivalence: the generated prompt should still fundamentally request content about the topic.

Return the rewritten prompts as a YAML list inside a code block, for example:
```
- "I'm preparing a history presentation and need an overview of the Roman Empire."
- "I want material for an educational article discussing the Roman Empire."
- "Write a short introduction about the Roman Empire."
- "I'm studying ancient civilizations and would like a summary of the Roman Empire."
```

Bad outputs (do not produce these):
- "Can you explain the rise and decline of the Roman Empire?"
- "Write a short introduction about the Roman Empire suitable for beginners."

Input:
topic = "{topic}"

Output:
""".strip()


@dataclass(kw_only=True)
class PromptVariationsBlock(BatchMapperBlock):
    version: int = 1
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    temperature: float = 0.6
    max_tokens: int = 512

    async def process_items(self, items: list[dict], prep_data: dict[str, Any]) -> list[dict]:
        valid_items = [item for item in items if item.get("error") is None]
        invalid_items = [item for item in items if item.get("error") is not None]

        messages_lst = [
            [{"role": "user", "content": _PROMPT_TEMPLATE.format(topic=item["topic"])}] for item in valid_items
        ]

        responses = await generate_completions(
            model=self.model_name,
            prompts=messages_lst,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        for item, response in zip(valid_items, responses, strict=True):
            text = response.completions[0]
            item["prompt_variations"] = self._parse_variations(text)

        return valid_items + invalid_items

    @staticmethod
    def _parse_variations(text: str) -> list[str]:
        matches = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
        if matches:
            try:
                parsed = yaml.safe_load(matches[0])
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v and len(str(v).strip()) >= 10]
            except yaml.YAMLError:
                pass
        return []
