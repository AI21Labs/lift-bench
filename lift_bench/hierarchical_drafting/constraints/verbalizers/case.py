from lift_bench.hierarchical_drafting.constraints.case import (
    CaseType,
    CaseTypeConstraint,
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


@register_constraint_verbalizer(CaseTypeConstraint)
class CaseTypeConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=CaseTypeConstraint)

    def _default_verbalization(
        self, constraint: CaseTypeConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        case_type = constraint.case_type
        if case_type == CaseType.UPPER:
            if verbalization_mode == VerbalizationMode.USER:
                return SingleTurnVerbalization(user_verbalization="All the letters in the text should be UPPERCASE")
            else:
                return SingleTurnVerbalization(system_verbalization="Your responses should always be in ALL-CAPS.")
        elif case_type == CaseType.LOWER:
            if verbalization_mode == VerbalizationMode.USER:
                return SingleTurnVerbalization(user_verbalization="All the letters in the text should be lowercase")
            else:
                return SingleTurnVerbalization(
                    system_verbalization="Your responses should always be in all lower-case letters, no capitalization whatsoever."
                )
        elif case_type == CaseType.TITLE:
            if verbalization_mode == VerbalizationMode.USER:
                return SingleTurnVerbalization(
                    user_verbalization="The text case should be Title (all words start with a capital letter)"
                )
            else:
                return SingleTurnVerbalization(
                    system_verbalization="Your responses should always be in Title-Case (all words start with a capital letter)."
                )
        else:
            raise RuntimeError("Unknown case type")
