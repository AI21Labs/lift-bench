from lift_bench.hierarchical_drafting.pipeline.task_verbalization.base import (
    TaskVerbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.registry import (
    register_task_verbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_header_constraints,
)

_PROMPT_TEMPLATE = """
{opening} Follow these constraints:
{verbalized_constraints}
""".strip()


@register_task_verbalizer("flat")
class FlatTaskVerbalizer(TaskVerbalizer):
    """Verbalizer for flat (IFEval-style) tasks — no document outline or section structure."""

    async def build_messages(
        self,
        topic: str,
        document_outline: str,
        header_to_constraints_dct: dict,
        use_strict_header_names: bool,
        verbalized_topic: str | None = None,
    ) -> list[dict]:
        opening = (verbalized_topic or f"Write a response about '{topic}'").rstrip(".")
        if opening[-1] not in "?!":
            opening += "."
        verbalized_constraints = self._verbalize_constraints(header_to_constraints_dct)
        prompt = _PROMPT_TEMPLATE.format(opening=opening, verbalized_constraints=verbalized_constraints)
        return [{"role": "user", "content": prompt}]

    def _verbalize_constraints(self, header_to_constraints_dct: dict[str, dict[str, list[str]]]) -> str:
        items = []
        for header_constraints_dct in header_to_constraints_dct.values():
            items.extend(f"* {constraint_str}" for constraint_str in render_header_constraints(header_constraints_dct))
        return "\n".join(items)
