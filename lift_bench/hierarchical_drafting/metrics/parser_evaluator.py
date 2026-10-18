from lift_bench.hierarchical_drafting.metrics.registry import register_metric


@register_metric("parser_evaluator")
async def parser_evaluator(completion, should_succeed: bool = True) -> float:
    """`should_succeed` as a score once the output parser has extracted the target; a failed parse scores 0.0.

    The extraction itself is the check: `with_output_parser` returns 0.0 before reaching here when the
    scoped section or artifact can't be found.
    """
    return float(should_succeed)
