import argparse
import logging
import sys
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliImplicitFlag,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

EvalType = Literal["artifact_exists", "light_critic"]

_FLAT_VERBALIZERS = ("flat", "ifeval_flat")
_CONFIG_SUFFIXES = (".yaml", ".yml")


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineInfraGroup(_Group):
    """Pipeline infra"""

    max_examples: int = Field(default=5, description="Optional, maximum number of generated examples")
    seed_topics_path: str = Field(
        default="data/wiki_vital_articles_20230601_titles.jsonl",
        description="JSONL whose first line maps row ids to the document topics under `title`",
    )
    output_dir: str = "drafting_synth_outputs"
    pipeline_name: str = "sanity_check"
    log_level: int = logging.INFO
    max_concurrency: int = 20
    chunk_size: int = 500
    min_success_threshold: float = 0.2
    train_size: Optional[int | float] = None
    test_size: Optional[int | float] = 1
    split_by_field: str = Field(
        default="example_id",
        description=(
            "Field to key the train/test split on via a stable content hash, so a record's split "
            "depends only on its own key and never shifts between runs (safe for ensembling). "
            "An int test_size is read as a target count over the fixed initial seed-population size. "
            "Pass an empty string to fall back to the legacy position-based sklearn shuffle split."
        ),
    )
    dry_run: CliImplicitFlag[bool] = False


class DocumentOutlineGroup(_Group):
    """Document outline"""

    document_outline_model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct", description="Used model for generating document outline"
    )
    document_outline_batch_size: int = Field(
        default=20,
        description="Number of batched LLM requests for document outline (just for optimization purposes)",
    )
    document_outline_max_tokens: int = 2048
    document_outline_temperature: float = 0.6
    doc_template_max_children: int = Field(default=3, description="Max number of children per section")
    doc_template_max_depth: int = Field(default=2, description="Max depth of each document")
    doc_template_verbalizer_name: Literal["headers", "contents"] = Field(
        default="headers", description="Document outline verbalizer name"
    )


class LLMJudgeConstraintGenerationGroup(_Group):
    """LLM-as-a-judge constraint generation"""

    content_constraints_model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct",
        description="Used model for generating LLM-as-a-judge content constraints for a given section",
    )
    content_constraints_max_tokens: int = 2048
    content_constraints_temperature: float = 0.6
    max_content_constraints: int = Field(default=2, description="Max number of content constraints per section")
    max_artifact_constraints: int = Field(default=2, description="Max number of artifact constraints per section")
    content_eval_type: EvalType = Field(
        default="light_critic",
        description="How to evaluate section's content: artifact_exists (only check that content exists in the section), or light_critic (an LLM judge checks that content satisfies a given criterion)",
    )
    artifact_eval_type: EvalType = Field(
        default="light_critic",
        description="How to evaluate section's artifact: artifact_exists (only check that the artifact exists in the section), or light_critic (an LLM judge checks that the artifact satisfies a given criterion)",
    )
    llm_judge_constraint_critic_model_name: str = Field(
        default="google/gemma-4-31B-it",
        description="Config param `model_name` of the LLM judge of the artifact constraints.",
    )
    llm_judge_constraint_critic_n: int = Field(
        default=1, description="Config param `n` of the artifact constraints' LLM judge"
    )
    llm_judge_constraint_critic_temperature: float = Field(
        default=0.6, description="Config param `temperature` of the artifact constraints' LLM judge"
    )
    llm_judge_constraint_critic_max_tokens: int = Field(
        default=2048, description="Config param `max_tokens` of the artifact constraints' LLM judge"
    )


class LLMJudgeConstraintSamplingGroup(_Group):
    """LLM-as-a-judge constraint sampling"""

    non_verified_max_sampled_constraints: int = Field(
        default=1, description='Max number of sampled constraints (use `-1` for "all constraints")'
    )
    non_verified_sample_per_header: CliImplicitFlag[bool] = Field(
        default=True, description="If set to True, sample non-verifiable constraints per section independently"
    )
    non_verified_max_sampled_headers: int = Field(default=3, description="Max number of sampled sections")
    non_verified_exact_sampling: CliImplicitFlag[bool] = Field(
        default=True,
        description="If set, sample exactly max_sampled_constraints non-verifiable constraints; otherwise sample between 1 and max_sampled_constraints",
    )


class VerifiableConstraintSamplingGroup(_Group):
    """Verifiable constraint sampling"""

    verified_max_sampled_constraints: int = Field(
        default=1, description='Max number of sampled constraints (use `-1` for "all constraints")'
    )
    verified_sample_per_header: CliImplicitFlag[bool] = Field(
        default=True, description="If set to True, sample verifiable constraints per section independently"
    )
    verified_max_sampled_headers: int = Field(default=3, description="Max number of sampled sections")
    verified_exact_sampling: CliImplicitFlag[bool] = Field(
        default=True,
        description="If set, sample exactly max_sampled_constraints verifiable constraints; otherwise sample between 1 and max_sampled_constraints",
    )
    keyword_generation_model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct",
        description="Model to use for contextual keyword generation (based on the section name)",
    )
    keyword_generation_max_retries: int = Field(
        default=3,
        description="Max num of retries to use for contextual keyword generation (based on the section name)",
    )
    use_strict_header_names: CliImplicitFlag[bool] = Field(
        default=True,
        description="If set to True, add lexical constraints checking that header names match the names specified in the outline",
    )


class IFBenchConstraintSamplingGroup(_Group):
    """IFBench verifiable constraint sampling"""

    use_ifbench_constraints: CliImplicitFlag[bool] = False
    ifbench_max_sampled_constraints: int = Field(default=1, description="Max number of sampled constraints")
    ifbench_custom_instructions_lst: list[str] = Field(
        default=[
            "copy:repeat_phrase",
            "count:lowercase_counting",
            "detectable_format:bigram_wrapping",
            "detectable_format:sentence_hyphens",
            "detectable_format:square_brackets",
            "first_word:first_word_answer",
            "first_word:first_word_sent",
            "keywords:no_adjacent_consecutive",
            "keywords:palindrome",
            "keywords:word_count_different_numbers",
            "keywords:word_once",
            "last_word:last_word_answer",
            "last_word:last_word_sent",
            "letters:letter_counting",
            "letters:letter_counting2",
            "paragraphs:paragraphs",
            "paragraphs:paragraphs2",
            "punctuation:punctuation_dot",
            "punctuation:punctuation_exclamation",
        ],
        description="IFTrain instruction ids to sample from.",
    )


class DraftingTaskGroup(_Group):
    """Drafting task"""

    task_verbalizer_name: str = Field(
        default="trailing_constraints",
        description="Name of the task verbalizer to use. Use 'random' to sample randomly from all registered ones.",
    )
    task_mode: Literal["structured", "flat"] = Field(
        default="structured",
        description=(
            "Task generation mode. 'structured' (default) produces hierarchical markdown documents with "
            "section-scoped constraints. 'flat' produces IFEval-style tasks: a plain prompt with "
            "document-level constraints and no required heading structure."
        ),
    )
    content_weight: float = Field(
        default=0.9,
        description=(
            "Weight assigned to content evaluations when aggregating scores "
            "(structure evaluations receive 1 - content_weight). "
            "Not relevant to task generation — does not affect the block output hash."
        ),
    )
    verbalize_topic: CliImplicitFlag[bool] = Field(
        default=False,
        description=(
            "When set, run PromptVariationsBlock to rewrite the topic into diverse natural-language "
            "openings before verbalization. Disabled by default."
        ),
    )
    naturalize_constraints: CliImplicitFlag[bool] = Field(
        default=False,
        description=(
            "When set, run ConstraintNaturalizationBlock to rephrase each constraint's rigid, templated "
            "text into a natural-sounding instruction via an LLM, preserving its exact meaning. Applies "
            "uniformly to every task_verbalizer_name. Disabled by default."
        ),
    )
    constraint_naturalization_model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct",
        description="Used model for naturalizing constraint text (only relevant when --naturalize_constraints is set)",
    )
    constraint_naturalization_max_tokens: int = 1024
    constraint_naturalization_temperature: float = 0.6
    constraint_naturalization_batch_size: int = Field(
        default=8, description="Number of constraints rephrased per LLM call; batches are naturalized concurrently."
    )


class Params(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_avoid_json=True, cli_use_class_docs_for_groups=True)

    config: Optional[str] = Field(
        default=None,
        description="Path to a YAML config file. Values override field defaults but are overridden by explicit CLI args.",
    )

    pipeline_infra: PipelineInfraGroup = Field(default_factory=PipelineInfraGroup)
    doc_outline: DocumentOutlineGroup = Field(default_factory=DocumentOutlineGroup)
    llm_judge_constraint_generation: LLMJudgeConstraintGenerationGroup = Field(
        default_factory=LLMJudgeConstraintGenerationGroup
    )
    llm_judge_constraint_sampling: LLMJudgeConstraintSamplingGroup = Field(
        default_factory=LLMJudgeConstraintSamplingGroup
    )
    verifiable_constraint_sampling: VerifiableConstraintSamplingGroup = Field(
        default_factory=VerifiableConstraintSamplingGroup
    )
    ifbench_constraint_sampling: IFBenchConstraintSamplingGroup = Field(default_factory=IFBenchConstraintSamplingGroup)
    drafting_task: DraftingTaskGroup = Field(default_factory=DraftingTaskGroup)

    @model_validator(mode="after")
    def _apply_flat_task_mode(self) -> "Params":
        if self.drafting_task.task_mode == "flat":
            self.doc_outline.doc_template_max_children = 0
            self.doc_outline.doc_template_max_depth = 0
            self.verifiable_constraint_sampling.use_strict_header_names = False
            if self.drafting_task.task_verbalizer_name not in _FLAT_VERBALIZERS:
                self.drafting_task.task_verbalizer_name = "ifeval_flat"
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003 (required by base signature)
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 (required by base signature)
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 (required by base signature)
        **kwargs,  # noqa: ARG003 (required by base signature)
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Sources earlier in the tuple win: explicit CLI args override the config file.
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            CliSettingsSource(
                settings_cls, cli_parse_args=True, cli_avoid_json=True, cli_use_class_docs_for_groups=True
            ),
        ]
        config_path = _config_path_from_argv()
        if config_path is not None:
            if not config_path.endswith(_CONFIG_SUFFIXES):
                raise ValueError(f"Unknown config file type '{config_path}', expected one of {_CONFIG_SUFFIXES}")
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_path))
        return tuple(sources)


def _config_path_from_argv() -> Optional[str]:
    """The config file has to be read before the CLI source, so pick `--config` out of argv on its own."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    known, _ = config_parser.parse_known_args(sys.argv[1:])
    return known.config


def parse_args() -> Params:
    return Params()


if __name__ == "__main__":
    print(parse_args())
