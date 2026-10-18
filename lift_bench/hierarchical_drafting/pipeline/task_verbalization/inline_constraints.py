from lift_bench.hierarchical_drafting.pipeline.task_verbalization.base import (
    TaskVerbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.registry import (
    register_task_verbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_header_constraints,
)

_PROMPT_TEMPLATE_DEFAULT = """
Write a markdown document about '{topic}' that has the following outline. Each section may include constraints that must be satisfied in that section:

{inline_outline}

{outline_header_names_instruction}
""".strip()

_PROMPT_TEMPLATE_WITH_VARIATION = """
{verbalized_topic}

Write it as a markdown document with the following outline. Each section may include constraints that must be satisfied in that section:

{inline_outline}

{outline_header_names_instruction}
""".strip()


@register_task_verbalizer("inline_constraints")
class InlineConstraintsTaskVerbalizer(TaskVerbalizer):
    async def build_messages(
        self,
        topic: str,
        document_outline: str,
        header_to_constraints_dct: dict,
        use_strict_header_names: bool,
        verbalized_topic: str | None = None,
    ) -> list[dict]:
        inline_outline = self._build_inline_outline(document_outline, header_to_constraints_dct)
        outline_header_names_instruction = ""
        if use_strict_header_names:
            outline_header_names_instruction = (
                "The names of the headers in the generated document should *exactly* match the ones in the outline."
            )
        if verbalized_topic:
            prompt = _PROMPT_TEMPLATE_WITH_VARIATION.format(
                verbalized_topic=verbalized_topic,
                inline_outline=inline_outline,
                outline_header_names_instruction=outline_header_names_instruction,
            )
        else:
            prompt = _PROMPT_TEMPLATE_DEFAULT.format(
                topic=topic,
                inline_outline=inline_outline,
                outline_header_names_instruction=outline_header_names_instruction,
            )
        return [{"role": "user", "content": prompt}]

    def _build_inline_outline(self, document_outline: str, header_to_constraints_dct: dict) -> str:
        result = []
        for line in document_outline.split("\n"):
            result.append(line)
            if not line.startswith("#"):
                continue
            header_name = line.lstrip("#").strip()
            if header_name not in header_to_constraints_dct:
                continue
            constraints_block = self._verbalize_header_constraints(header_to_constraints_dct[header_name])
            if constraints_block:
                result.append(constraints_block)
        return "\n".join(result)

    def _verbalize_header_constraints(self, header_constraints_dct: dict[str, list[str]]) -> str:
        items = [f"  * {constraint_str}" for constraint_str in render_header_constraints(header_constraints_dct)]
        if not items:
            return ""
        return "  Constraints:\n" + "\n".join(items)
