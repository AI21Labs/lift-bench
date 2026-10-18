from dataclasses import dataclass
from typing import Optional

from lift_bench.hierarchical_drafting.constraints.base import (
    Constraint,
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


@polymorphic_childclass(type_field_value="emoji_count")
@dataclass(kw_only=True)
class EmojiCountConstraint(Constraint):
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    def __post_init__(self):
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("must provide either a lower bound, an upper bound or both, but both are None")
        if self.lower_bound is not None and self.lower_bound < 0:
            raise ValueError("lower_bound must be non-negative")
        if self.upper_bound is not None and self.upper_bound < 0:
            raise ValueError("upper_bound must be non-negative")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("Lower bound must be smaller or equal to upper bound")


@polymorphic_childclass(type_field_value="verifiable_emoji_count_constraint")
@dataclass(kw_only=True)
class VerifiableEmojiCountConstraint(EmojiCountConstraint, VerifiableConstraint):
    def to_evaluation_dict(self):
        return {
            "metric_name": "emoji_count",
            "metric_args": {
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
            },
        }
