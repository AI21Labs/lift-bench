import emoji

from lift_bench.hierarchical_drafting.metrics.registry import register_metric
from lift_bench.hierarchical_drafting.metrics.errors import MetricError


def _check_bound_validity(value: int, lower_bound: int | None, upper_bound: int | None) -> bool:
    if lower_bound is None:
        return value <= upper_bound
    if upper_bound is None:
        return value >= lower_bound
    return lower_bound <= value <= upper_bound


@register_metric("emoji_count")
async def emoji_count(
    completion,
    lower_bound: int | None,
    upper_bound: int | None,
) -> float:
    if lower_bound is None and upper_bound is None:
        raise MetricError("emoji_count needs a lower or an upper bound")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise MetricError(f"emoji_count's lower bound {lower_bound} exceeds its upper bound {upper_bound}")
    text = completion[-1]["content"]
    count = emoji.emoji_count(string=text, unique=False)
    is_valid = _check_bound_validity(count, lower_bound, upper_bound)
    return 1.0 if is_valid else 0.0
