from lift_bench.hierarchical_drafting.constraints.artifact import (
    ArtifactConstraint,
    ArtifactExistanceConstraint,
    ArtifactLLMJudgeConstraint,
)
from lift_bench.hierarchical_drafting.constraints.registry import (
    register_constraint_verbalizer,
)
from lift_bench.hierarchical_drafting.constraints.verbalization_mode import (
    SingleTurnVerbalization,
)


@register_constraint_verbalizer(ArtifactLLMJudgeConstraint)
@register_constraint_verbalizer(ArtifactExistanceConstraint)
class AbsoluteCriticConstraintVerbalizer:
    def verbalize(self, constraint: ArtifactConstraint) -> SingleTurnVerbalization:
        return SingleTurnVerbalization(user_verbalization=constraint.criterion)
