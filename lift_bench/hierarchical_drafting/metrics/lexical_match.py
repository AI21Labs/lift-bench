from lift_bench.hierarchical_drafting.metrics._lexical_utils import (
    MatchType,
    lexical_match_count,
)
from lift_bench.hierarchical_drafting.metrics.registry import register_metric


@register_metric("lexical_match")
async def lexical_match(
    completion,
    string: str,
    match_type: str,
    should_match: bool,
    fuzzy_match: bool,
) -> float:
    match_type = MatchType(match_type)
    text = completion[-1]["content"]
    count = lexical_match_count(text, string, match_type, fuzzy_match)
    is_match = count > 0
    is_valid = is_match if should_match else not is_match
    return 1.0 if is_valid else 0.0
