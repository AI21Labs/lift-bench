from lift_bench.hierarchical_drafting.pipeline.task_verbalization.base import (
    TaskVerbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.registry import (
    register_task_verbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_header_constraints,
)

# Used when no verbalized_topic is provided.
_PROMPT_TEMPLATE_DEFAULT = """
Write a markdown document about '{topic}' that has the following outline:
{document_outline}

{outline_header_names_instruction}In addition, follow the constraints below:
{verbalized_constraints}
""".strip()

# Used when a topic variation is available — the variation replaces the opening instruction.
_PROMPT_TEMPLATE_WITH_VARIATION = """
{verbalized_topic}

Write it as a markdown document with the following outline:
{document_outline}

{outline_header_names_instruction}In addition, follow the constraints below:
{verbalized_constraints}
""".strip()


@register_task_verbalizer("trailing_constraints")
class DefaultTaskVerbalizer(TaskVerbalizer):
    async def build_messages(
        self,
        topic: str,
        document_outline: str,
        header_to_constraints_dct: dict,
        use_strict_header_names: bool,
        verbalized_topic: str | None = None,
    ) -> list[dict]:
        verbalized_constraints = self._verbalize_constraints(header_to_constraints_dct)
        outline_header_names_instruction = ""
        if use_strict_header_names:
            outline_header_names_instruction = (
                "The names of the headers in the generated document should *exactly* match the ones in the outline.\n"
            )
        if verbalized_topic:
            prompt = _PROMPT_TEMPLATE_WITH_VARIATION.format(
                verbalized_topic=verbalized_topic,
                document_outline=document_outline,
                outline_header_names_instruction=outline_header_names_instruction,
                verbalized_constraints=verbalized_constraints,
            )
        else:
            prompt = _PROMPT_TEMPLATE_DEFAULT.format(
                topic=topic,
                document_outline=document_outline,
                outline_header_names_instruction=outline_header_names_instruction,
                verbalized_constraints=verbalized_constraints,
            )
        return [{"role": "user", "content": prompt}]

    def _verbalize_constraints(self, header_to_constraints_dct: dict[str, dict[str, list[str]]]) -> str:
        s = ""
        for header_name, header_constraints_dct in header_to_constraints_dct.items():
            rendered = render_header_constraints(header_constraints_dct)
            if not rendered:
                continue
            curr_header_constraints_str = "".join(f"\t* {constraint_str}\n" for constraint_str in rendered)
            s += f"The '{header_name}' section should follow the constraints below:\n"
            s += curr_header_constraints_str + "\n"
        return s
