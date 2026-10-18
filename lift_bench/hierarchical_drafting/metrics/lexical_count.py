from lift_bench.hierarchical_drafting.metrics._lexical_utils import (
    MatchType,
    check_bound_validity,
    lexical_match_count,
)
from lift_bench.hierarchical_drafting.metrics.registry import register_metric


@register_metric("lexical_count")
async def lexical_count(
    completion,
    string: str,
    lower_bound: int | None,
    upper_bound: int | None,
    fuzzy_match: bool,
    strict_word_boundaries: bool,
) -> float:
    text = completion[-1]["content"]
    count = lexical_match_count(text, string, MatchType.CONTAINS, fuzzy_match, strict_word_boundaries)
    is_valid = check_bound_validity(count, lower_bound, upper_bound)
    return 1.0 if is_valid else 0.0
