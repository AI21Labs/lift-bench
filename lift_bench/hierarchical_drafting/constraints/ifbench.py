from dataclasses import dataclass, field

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


@polymorphic_childclass(type_field_value="ifbench_constraint")
@dataclass(kw_only=True)
class IFBenchConstraint(VerifiableConstraint):
    """Base class for IFBench constraints with instruction_id and instruction_kwargs."""

    instruction_kwargs: dict[str, ...] = field(default_factory=dict)
    instruction_id: str

    def __post_init__(self):
        if self.instruction_kwargs is None:
            self.instruction_kwargs = {}

    def to_evaluation_dict(self):
        return {
            "metric_name": "ifbench",
            "metric_args": {
                "instruction_id_list": [self.instruction_id],
                "kwargs": [self.instruction_kwargs],
            },
        }
