from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from lift_bench.hierarchical_drafting.constraints.base import (
    Constraint,
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


class MatchType(StrEnum):
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    EXACT = "exact"
    REGEX = "regex"


_LEXICAL_VERSION = 2


@polymorphic_childclass(type_field_value="lexical")
@dataclass(kw_only=True)
class LexicalConstraint(Constraint):
    version: int = _LEXICAL_VERSION
    string: str
    match_type: MatchType
    should_match: bool = True
    fuzzy_match: bool = True

    def __post_init__(self):
        if len(self.string) == 0:
            raise ValueError("must provide a non-empty string for matching")

    @classmethod
    def preprocess_dict(cls, data):
        version = data.get("version", 1)
        if version < 2:
            data["fuzzy_match"] = not data.pop("case_sensitive", False)
            del data["strict_word_boundaries"]
        data["version"] = _LEXICAL_VERSION


@polymorphic_childclass(type_field_value="lexical_count")
@dataclass(kw_only=True)
class LexicalCountConstraint(Constraint):
    string: str
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    fuzzy_match: bool = True
    strict_word_boundaries: bool = True

    def __post_init__(self):
        if len(self.string) == 0:
            raise ValueError("must provide a non-empty string for matching")
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("must provide either a lower bound, an upper bound or both, but both are None")
        if (self.lower_bound is None or self.lower_bound < 0) and (self.upper_bound is None or self.upper_bound < 0):
            raise ValueError("must provide a non-negative appearance count")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("Lower bound must be smaller or equal to upper bound")


@polymorphic_childclass(type_field_value="verifiable_lexical_count_constraint")
@dataclass(kw_only=True)
class VerifiableLexicalCountConstraint(LexicalCountConstraint, VerifiableConstraint):
    def to_evaluation_dict(self):
        return {
            "metric_name": "lexical_count",
            "metric_args": {
                "string": self.string,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "fuzzy_match": self.fuzzy_match,
                "strict_word_boundaries": self.strict_word_boundaries,
            },
        }


@polymorphic_childclass(type_field_value="verifiable_lexical_constraint")
@dataclass(kw_only=True)
class VerifiableLexicalConstraint(LexicalConstraint, VerifiableConstraint):
    def to_evaluation_dict(self):
        return {
            "metric_name": "lexical_match",
            "metric_args": {
                "string": self.string,
                "match_type": self.match_type,
                "should_match": self.should_match,
                "fuzzy_match": self.fuzzy_match,
            },
        }
