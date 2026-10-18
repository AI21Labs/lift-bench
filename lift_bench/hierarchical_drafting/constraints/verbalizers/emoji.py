from lift_bench.hierarchical_drafting.constraints.emoji import (
    EmojiCountConstraint,
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


def _is_single(lower_bound, upper_bound):
    if lower_bound is None:
        return upper_bound == 1
    if upper_bound is None:
        return lower_bound == 1
    if upper_bound == lower_bound:
        return lower_bound == 1
    return False


def _get_range_type_verbalization_emoji(lower_bound, upper_bound):
    if lower_bound is None:
        return f"at most {upper_bound}" if upper_bound > 0 else "no"
    if upper_bound is None:
        return f"at least {lower_bound}"
    if upper_bound == lower_bound:
        return f"exactly {lower_bound}" if lower_bound > 0 else "no"
    return f"between {lower_bound} and {upper_bound}"


@register_constraint_verbalizer(EmojiCountConstraint)
class EmojiCountConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=EmojiCountConstraint)

    def _default_verbalization(
        self, constraint: EmojiCountConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        plural = "s" if not _is_single(constraint.lower_bound, constraint.upper_bound) else ""
        range_verb = _get_range_type_verbalization_emoji(constraint.lower_bound, constraint.upper_bound)
        if verbalization_mode == VerbalizationMode.USER:
            template = f"The text should contain {range_verb} emoji{plural}."
            return SingleTurnVerbalization(user_verbalization=template)
        else:
            template = f"Your responses should always contain {range_verb} emoji{plural}."
            return SingleTurnVerbalization(system_verbalization=template)
