from dataclasses import dataclass

from lift_bench.hierarchical_drafting.constraints.base import (
    Constraint,
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


@polymorphic_childclass(type_field_value="language")
@dataclass(kw_only=True)
class TargetLanguageConstraint(Constraint):
    language_code: str


@polymorphic_childclass(type_field_value="verifiable_language_constraint")
@dataclass(kw_only=True)
class VerifiableTargetLanguageConstraint(TargetLanguageConstraint, VerifiableConstraint):
    def to_evaluation_dict(self):
        return {
            "metric_name": "target_language",
            "metric_args": {
                "language_code": self.language_code,
            },
        }
