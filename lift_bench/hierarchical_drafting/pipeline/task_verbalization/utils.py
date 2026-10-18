from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.constraints.registry import (
    get_constraint_verbalizer,
)


def render_constraint_text(constraint_metadata: VerifiableConstraintMetadata) -> str:
    """Render a single constraint's prompt text, preferring a naturalized override if one was set."""
    if constraint_metadata.custom_text:
        return constraint_metadata.custom_text
    constraint = constraint_metadata.constraint
    verbalizer = get_constraint_verbalizer(constraint)
    return verbalizer.verbalize(constraint).to_prompt()


def render_header_constraints(header_constraints_dct: dict[str, list[str]]) -> list[str]:
    """Render every non-structural constraint for a single header into its prompt text."""
    rendered = []
    for constraint_type, constraint_json_str_lst in header_constraints_dct.items():
        if constraint_type == "doc_structure":
            continue
        for constraint_json_str in constraint_json_str_lst:
            constraint_metadata = VerifiableConstraintMetadata.from_json(constraint_json_str)
            rendered.append(render_constraint_text(constraint_metadata))
    return rendered
