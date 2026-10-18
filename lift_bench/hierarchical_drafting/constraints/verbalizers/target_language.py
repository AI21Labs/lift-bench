from lift_bench.hierarchical_drafting.constraints.languages import (
    LANGCODE2LANGUAGE,
)
from lift_bench.hierarchical_drafting.constraints.registry import (
    register_constraint_verbalizer,
)
from lift_bench.hierarchical_drafting.constraints.target_language import (
    TargetLanguageConstraint,
)
from lift_bench.hierarchical_drafting.constraints.verbalization_mode import (
    SingleTurnVerbalization,
    VerbalizationMode,
)
from lift_bench.hierarchical_drafting.constraints.verbalizers.base import (
    ConstraintVerbalizer,
)


@register_constraint_verbalizer(TargetLanguageConstraint)
class TargetLanguageConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=TargetLanguageConstraint)

    def _default_verbalization(
        self, constraint: TargetLanguageConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        if verbalization_mode == VerbalizationMode.USER:
            text = f"The text should be written in {LANGCODE2LANGUAGE[constraint.language_code]}."
            return SingleTurnVerbalization(user_verbalization=text)
        else:
            text = f"The responses should always be written in {LANGCODE2LANGUAGE[constraint.language_code]}."
            return SingleTurnVerbalization(system_verbalization=text)
