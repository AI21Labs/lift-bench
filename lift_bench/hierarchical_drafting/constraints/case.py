from dataclasses import dataclass
from enum import StrEnum

from lift_bench.hierarchical_drafting.constraints.base import (
    Constraint,
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


class CaseType(StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    TITLE = "title"  # First letter of each word is capitalized


@polymorphic_childclass(type_field_value="case_type")
@dataclass(kw_only=True)
class CaseTypeConstraint(Constraint):
    case_type: CaseType


@polymorphic_childclass(type_field_value="verifiable_case_type_constraint")
@dataclass(kw_only=True)
class VerifiableCaseTypeConstraint(CaseTypeConstraint, VerifiableConstraint):
    def to_evaluation_dict(self):
        return {
            "metric_name": "case_type",
            "metric_args": {
                "case_type": self.case_type,
            },
        }
