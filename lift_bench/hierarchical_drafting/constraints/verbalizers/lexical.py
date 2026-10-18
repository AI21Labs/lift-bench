import string

from lift_bench.hierarchical_drafting.constraints.lexical import (
    LexicalConstraint,
    LexicalCountConstraint,
    MatchType,
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


def _get_range_type_verbalization_count(lower_bound, upper_bound):
    if lower_bound is None:
        return f"at most {upper_bound} times" if upper_bound > 1 else "at most once"
    if upper_bound is None:
        return f"at least {lower_bound} times" if lower_bound > 1 else "at least once"
    if upper_bound == lower_bound:
        return f"exactly {lower_bound} times" if lower_bound > 1 else "exactly once"
    return f"between {lower_bound} and {upper_bound} times"


def _get_string_type_verbalization(s, strict_word_boundaries):
    if len(s) == 1:
        return f"the word '{s}'" if strict_word_boundaries else f"the character '{s}'"
    if not any(ws in s for ws in string.whitespace) and strict_word_boundaries:
        return f"the word '{s}'"
    return f"the string '{s}'"


@register_constraint_verbalizer(LexicalConstraint)
class LexicalConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=LexicalConstraint)

    def _default_verbalization(
        self, constraint: LexicalConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        match_type = self._match_type_verbalization(match_type=constraint.match_type)
        negation_addition = " NOT" if not constraint.should_match else ""
        if verbalization_mode == VerbalizationMode.USER:
            prefix = f"The text should{negation_addition}"
        else:
            prefix = "Your responses should always" if not negation_addition else "Your responses should NEVER"
        template = f'{prefix} {match_type} "{constraint.string}"'
        if verbalization_mode == VerbalizationMode.USER:
            return SingleTurnVerbalization(user_verbalization=template)
        else:
            return SingleTurnVerbalization(system_verbalization=template)

    @staticmethod
    def _match_type_verbalization(match_type: MatchType) -> str:
        if match_type == MatchType.CONTAINS:
            return "contain the phrase"
        elif match_type == MatchType.STARTS_WITH:
            return "start with the phrase"
        elif match_type == MatchType.ENDS_WITH:
            return "end with the phrase"
        elif match_type == MatchType.EXACT:
            return "be exactly"
        else:
            raise ValueError(f"Unsupported match type: {match_type}")


@register_constraint_verbalizer(LexicalCountConstraint)
class LexicalCountConstraintVerbalizer(ConstraintVerbalizer):
    def __init__(self):
        super().__init__(supported_constraint=LexicalCountConstraint)

    def _default_verbalization(
        self, constraint: LexicalCountConstraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        mode_prefix = (
            "The text should contain"
            if verbalization_mode == VerbalizationMode.USER
            else "Your responses should always contain"
        )
        template = (
            f"{mode_prefix} {_get_string_type_verbalization(constraint.string, constraint.strict_word_boundaries)} "
            f"{_get_range_type_verbalization_count(constraint.lower_bound, constraint.upper_bound)}."
        )
        if verbalization_mode == VerbalizationMode.USER:
            return SingleTurnVerbalization(user_verbalization=template)
        else:
            return SingleTurnVerbalization(system_verbalization=template)
