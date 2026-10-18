import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_constraint_text,
)
from lift_bench.hierarchical_drafting.utils.inference import (
    generate_completions,
)

_NATURALIZE_PROMPT_TEMPLATE = """
Rewrite each of the constraints below so it reads as a natural instruction, instead of a rigid templated one (e.g. avoid always starting with "The text should ..."). Preserve every constraint's meaning exactly: do not drop, soften, or add any requirement. Keep them as {n} independent, standalone items in the same order — do not merge them into a single sentence or paragraph.

Return the rewritten items as a YAML list inside a markdown code block, for example:
```
- Rewritten first constraint...
- Rewritten second constraint...
```

Constraints:
{numbered_constraints}
""".strip()


@dataclass(kw_only=True)
class ConstraintNaturalizationBlock(MapperBlock):
    """Rephrases each constraint's rigid, templated text into a natural-sounding instruction via an LLM, preserving
    its exact meaning. Constraints are batched (across headers/constraint types) so each LLM call stays small, with
    batches naturalized concurrently. Meant to run right before constraint sampling (on the full generated
    constraint pool, before any max_sampled_constraints/max_sampled_headers subsetting) rather than after: since
    sampling sweeps typically vary only the sampling knobs while the upstream constraint pool stays identical, this
    lets the block's own output caching (keyed on its inputs_path/attributes_hash) skip recomputation entirely
    across sampling variants, instead of re-naturalizing a different sampled subset from scratch each time. Every
    existing task verbalizer picks up the naturalized text automatically via VerifiableConstraintMetadata.custom_text,
    so no verbalizer-specific wiring is needed. Batches that fail to parse fall back to the original rigid text."""

    version: int = 1
    constraint_columns: list[str] = field(default_factory=lambda: ["llm_judge_constraints", "verifiable_constraints"])
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    temperature: float = 0.6
    max_tokens: int = 1024
    batch_size: int = 8

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        refs = []  # (constraint_col, header_name, constraint_type, index, metadata)
        rigid_texts = []
        for col in self.constraint_columns:
            header_to_constraints_dct = item.get(col)
            if not header_to_constraints_dct:
                continue
            for header_name, header_constraints_dct in header_to_constraints_dct.items():
                for constraint_type, constraint_json_str_lst in header_constraints_dct.items():
                    if constraint_type == "doc_structure":
                        continue
                    for index, constraint_json_str in enumerate(constraint_json_str_lst):
                        metadata = VerifiableConstraintMetadata.from_json(constraint_json_str)
                        refs.append((col, header_name, constraint_type, index, metadata))
                        rigid_texts.append(render_constraint_text(metadata))

        if not refs:
            return item

        batched_refs = [refs[i : i + self.batch_size] for i in range(0, len(refs), self.batch_size)]
        batched_texts = [rigid_texts[i : i + self.batch_size] for i in range(0, len(rigid_texts), self.batch_size)]
        prompts = [[{"role": "user", "content": self._build_prompt(texts)}] for texts in batched_texts]

        responses = await generate_completions(
            model=self.model_name,
            prompts=prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        n_naturalized = 0
        n_fallback = 0
        for batch_refs, batch_texts, response in zip(batched_refs, batched_texts, responses):
            response_text = response.completions[0] if response.completions else ""
            rephrased = self._parse_response(response_text, len(batch_texts))
            if rephrased is None:
                n_fallback += len(batch_refs)
                continue
            for (col, header_name, constraint_type, index, metadata), rephrased_text in zip(batch_refs, rephrased):
                metadata.custom_text = rephrased_text
                item[col][header_name][constraint_type][index] = metadata.to_json()
            n_naturalized += len(batch_refs)

        if n_fallback:
            self.logger.warning(
                f"Constraint naturalization: {n_fallback}/{n_naturalized + n_fallback} constraints fell back to "
                f"rigid template text for item {item.get('example_id', '?')} (unparseable/malformed LLM response)."
            )
        return item

    def _build_prompt(self, batch_texts: list[str]) -> str:
        numbered_constraints = "\n".join(f"{i}. {text}" for i, text in enumerate(batch_texts, start=1))
        return _NATURALIZE_PROMPT_TEMPLATE.format(n=len(batch_texts), numbered_constraints=numbered_constraints)

    @staticmethod
    def _parse_response(text: str, expected_count: int) -> Optional[list[str]]:
        matches = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
        if not matches:
            return None
        try:
            parsed = yaml.safe_load(matches[0])
        except yaml.YAMLError:
            return None
        if not (isinstance(parsed, list) and len(parsed) == expected_count):
            return None
        rephrased = [str(v).strip() for v in parsed]
        if any(not v for v in rephrased):
            return None
        return rephrased
