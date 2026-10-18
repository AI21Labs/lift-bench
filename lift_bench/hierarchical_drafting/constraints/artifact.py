from dataclasses import dataclass, field

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraint,
)
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_childclass,
)


@polymorphic_childclass(type_field_value="artifact")
@dataclass(kw_only=True)
class ArtifactConstraint(VerifiableConstraint):
    criterion: str


@polymorphic_childclass(type_field_value="artifact_llm_judge")
@dataclass(kw_only=True)
class ArtifactLLMJudgeConstraint(ArtifactConstraint):
    """
    An LLM judge assesses how much the text is aligned with the criterion.
    """

    critic_model_name: str = "google/gemma-4-31B-it"
    n: int = 1
    temperature: int = 0.6
    max_tokens: int = 2048
    custom_user_prompt: str = None
    ignore_criteria: bool = False

    def __post_init__(self):
        assert (not self.ignore_criteria) or self.custom_user_prompt is not None, (
            "Can't ignore criteria when not using a custom prompt"
        )

    def to_evaluation_dict(self):
        metric_args = {
            "critic_model_name": self.critic_model_name,
            "n": self.n,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_binary_range": False,  # the light critic defaults to a binary score range
        }
        if not self.ignore_criteria:
            metric_args["criteria"] = [self.criterion]
        if self.custom_user_prompt is not None:
            metric_args["messages"] = [{"role": "user", "content": self.custom_user_prompt}]

        return {
            "metric_name": "light_absolute_critic",
            "metric_args": metric_args,
        }


@polymorphic_childclass(type_field_value="artifact_no_llm_judge")
@dataclass(kw_only=True)
class ArtifactExistanceConstraint(ArtifactConstraint):
    """
    Simply check that a response was provided (if the parsing doesn't fail, it means a relevant text was provided).
    """

    should_succeed: bool = field()

    def to_evaluation_dict(self):
        return {
            "metric_name": "parser_evaluator",
            "metric_args": {
                "should_succeed": self.should_succeed,
            },
        }
