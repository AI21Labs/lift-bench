from lift_bench.hierarchical_drafting.iftrain.instructions_registry import (
    INSTRUCTION_DICT,
)

from lift_bench.hierarchical_drafting.constraints.ifbench import (
    IFBenchConstraint,
)
from lift_bench.hierarchical_drafting.constraints.registry import (
    register_constraint_verbalizer,
)
from lift_bench.hierarchical_drafting.constraints.verbalization_mode import (
    SingleTurnVerbalization,
    VerbalizationMode,
)
from lift_bench.hierarchical_drafting.constraints.verbalizers.base import (
    ConstraintVerbalizer,
)


@register_constraint_verbalizer(IFBenchConstraint)
class IFBenchConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=IFBenchConstraint)

    def _default_verbalization(
        self, constraint: IFBenchConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        instruction_cls = INSTRUCTION_DICT[constraint.instruction_id]
        instruction = instruction_cls(constraint.instruction_id)
        verbalized_instruction = instruction.build_description(**constraint.instruction_kwargs)
        return SingleTurnVerbalization(user_verbalization=verbalized_instruction)
