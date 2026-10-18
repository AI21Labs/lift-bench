import asyncio
import logging

from lift_bench.hierarchical_drafting.pipeline.blocks.constraint_aggregation import (
    ConstraintTypeAggregationBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.utils_blocks import (
    TrainTestSplitBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.case import (
    CaseTypeConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.emoji_count import (
    EmojiCountConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.iftrain.block import (
    IFBenchConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.lexical_count import (
    LexicalCountConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.target_language import (
    TargetLanguageConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.cli import (
    Params,
    parse_args,
)
import os
from lift_bench.hierarchical_drafting.core.base_blocks import (
    AtomicBlock,
)

from lift_bench.hierarchical_drafting.core.utils_blocks import (
    CombineDatasetsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.llm_judge.code import (
    CodeblockConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.llm_judge.content import (
    ContentConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.document_outline import (
    DocumentOutlineBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.document_template import (
    DocumentOutlineTemplateBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.drafting_task import (
    DraftingTaskBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.constraint_naturalization import (
    ConstraintNaturalizationBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.prompt_variations import (
    PromptVariationsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.seed_config import (
    GenerateSeedDraftingConfigBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.llm_judge.list import (
    ListConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.constraint_sampling import (
    ConstraintSamplingBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.seed_topics import (
    SeedTopicsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.llm_judge.table import (
    TableConstraintsBlock,
)
from lift_bench.hierarchical_drafting.pipeline.blocks.verifiable.header_names import (
    HeaderNameConstraintsBlock,
)
from lift_bench.hierarchical_drafting.utils.file_utils import dump_json, iter_json_lines
from lift_bench.hierarchical_drafting.pipeline.analytics import (
    visualize_data,
)


async def drafting_synth_pipeline(params: Params) -> dict[str, str | int]:
    logging.getLogger(params.pipeline_infra.pipeline_name).info("Starting drafting synthesis pipeline")
    pipeline_params = {
        "output_dir": params.pipeline_infra.output_dir,
        "pipeline_name": params.pipeline_infra.pipeline_name,
        "log_level": params.pipeline_infra.log_level,
        "dry_run": params.pipeline_infra.dry_run,
    }
    mapper_params = {
        "max_concurrency": params.pipeline_infra.max_concurrency,
        "chunk_size": params.pipeline_infra.chunk_size,
        "min_success_threshold": params.pipeline_infra.min_success_threshold,
    }
    artifact_llm_judge_critic_params = {
        "critic_model_name": params.llm_judge_constraint_generation.llm_judge_constraint_critic_model_name,
        "n": params.llm_judge_constraint_generation.llm_judge_constraint_critic_n,
        "temperature": params.llm_judge_constraint_generation.llm_judge_constraint_critic_temperature,
        "max_tokens": params.llm_judge_constraint_generation.llm_judge_constraint_critic_max_tokens,
    }

    last_block: AtomicBlock  # pointer to last block in the pipeline

    # ***************************************
    # ****** Generate document outline ******
    # ***************************************
    last_block = seed_config = GenerateSeedDraftingConfigBlock(
        seed_method="wiki_topics",
        topics_path=params.pipeline_infra.seed_topics_path,
        **pipeline_params,
    )
    await seed_config.run()

    last_block = seed_topics = SeedTopicsBlock(
        config_path=last_block.output_path,
        max_items=params.pipeline_infra.max_examples,
        **pipeline_params,
    )
    await seed_topics.run()

    last_block = doc_templates = DocumentOutlineTemplateBlock(
        max_children=params.doc_outline.doc_template_max_children,
        max_depth=params.doc_outline.doc_template_max_depth,
        template_verbalizer_name=params.doc_outline.doc_template_verbalizer_name,
        inputs_path=last_block.output_path,
        **pipeline_params,
        **mapper_params,
    )
    await doc_templates.run()

    last_block = doc_outlines = DocumentOutlineBlock(
        inputs_path=last_block.output_path,
        batch_size=params.doc_outline.document_outline_batch_size,
        model_name=params.doc_outline.document_outline_model_name,
        max_tokens=params.doc_outline.document_outline_max_tokens,
        temperature=params.doc_outline.document_outline_temperature,
        template_verbalizer_name=params.doc_outline.doc_template_verbalizer_name,
        **pipeline_params,
        **mapper_params,
    )
    await doc_outlines.run()

    feature_blocks = []

    # TODO: implement this
    assert (
        params.llm_judge_constraint_generation.max_content_constraints > 0
        or params.llm_judge_constraint_generation.max_artifact_constraints > 0
    ), "Case with no content/artifacts is not supported right now"

    # ****************************************
    # ****** Non-verifiable constraints ******
    # ****************************************
    if params.llm_judge_constraint_generation.max_content_constraints > 0:
        last_block = content_constraints = ContentConstraintsBlock(
            inputs_path=doc_outlines.output_path,
            max_constraints=params.llm_judge_constraint_generation.max_content_constraints,
            model_name=params.llm_judge_constraint_generation.content_constraints_model_name,
            max_tokens=params.llm_judge_constraint_generation.content_constraints_max_tokens,
            temperature=params.llm_judge_constraint_generation.content_constraints_temperature,
            eval_type=params.llm_judge_constraint_generation.content_eval_type,
            llm_judge_critic_params=artifact_llm_judge_critic_params,
            **pipeline_params,
            **mapper_params,
        )
        await content_constraints.run()
        feature_blocks.append(content_constraints)

    if params.llm_judge_constraint_generation.max_artifact_constraints > 0:
        # TODO: add ToneConstraintsBlock
        last_block = code_constraints = CodeblockConstraintsBlock(
            inputs_path=doc_outlines.output_path,
            model_name=params.llm_judge_constraint_generation.content_constraints_model_name,
            max_tokens=params.llm_judge_constraint_generation.content_constraints_max_tokens,
            temperature=params.llm_judge_constraint_generation.content_constraints_temperature,
            max_artifacts=params.llm_judge_constraint_generation.max_artifact_constraints,
            eval_type=params.llm_judge_constraint_generation.artifact_eval_type,
            llm_judge_critic_params=artifact_llm_judge_critic_params,
            **pipeline_params,
            **mapper_params,
        )
        await code_constraints.run()
        feature_blocks.append(code_constraints)

        last_block = table_constraints = TableConstraintsBlock(
            inputs_path=doc_outlines.output_path,
            model_name=params.llm_judge_constraint_generation.content_constraints_model_name,
            max_tokens=params.llm_judge_constraint_generation.content_constraints_max_tokens,
            temperature=params.llm_judge_constraint_generation.content_constraints_temperature,
            max_artifacts=params.llm_judge_constraint_generation.max_artifact_constraints,
            eval_type=params.llm_judge_constraint_generation.artifact_eval_type,
            llm_judge_critic_params=artifact_llm_judge_critic_params,
            **pipeline_params,
            **mapper_params,
        )
        await table_constraints.run()
        feature_blocks.append(table_constraints)

        last_block = list_constraints = ListConstraintsBlock(
            inputs_path=doc_outlines.output_path,
            model_name=params.llm_judge_constraint_generation.content_constraints_model_name,
            max_tokens=params.llm_judge_constraint_generation.content_constraints_max_tokens,
            temperature=params.llm_judge_constraint_generation.content_constraints_temperature,
            max_artifacts=params.llm_judge_constraint_generation.max_artifact_constraints,
            eval_type=params.llm_judge_constraint_generation.artifact_eval_type,
            llm_judge_critic_params=artifact_llm_judge_critic_params,
            **pipeline_params,
            **mapper_params,
        )
        await list_constraints.run()
        feature_blocks.append(list_constraints)

    # ************************************
    # ****** Verifiable constraints ******
    # ************************************
    if params.ifbench_constraint_sampling.use_ifbench_constraints:
        last_block = ifbench_constraints = IFBenchConstraintsBlock(
            inputs_path=doc_outlines.output_path,
            custom_instructions_lst=params.ifbench_constraint_sampling.ifbench_custom_instructions_lst,
            max_sampled_constraints=params.ifbench_constraint_sampling.ifbench_max_sampled_constraints,
            **pipeline_params,
            **mapper_params,
        )
        await ifbench_constraints.run()
        feature_blocks.append(ifbench_constraints)

    last_block = lang_constraints = TargetLanguageConstraintsBlock(
        inputs_path=doc_outlines.output_path,
        **pipeline_params,
        **mapper_params,
    )
    await lang_constraints.run()
    feature_blocks.append(lang_constraints)

    last_block = case_type_constraints = CaseTypeConstraintsBlock(
        inputs_path=doc_outlines.output_path,
        **pipeline_params,
        **mapper_params,
    )
    await case_type_constraints.run()
    feature_blocks.append(case_type_constraints)

    last_block = lexical_count_constraints = LexicalCountConstraintsBlock(
        inputs_path=doc_outlines.output_path,
        keyword_generation_model_name=params.verifiable_constraint_sampling.keyword_generation_model_name,
        keyword_generation_max_retries=params.verifiable_constraint_sampling.keyword_generation_max_retries,
        **pipeline_params,
        **mapper_params,
    )
    await lexical_count_constraints.run()
    feature_blocks.append(lexical_count_constraints)

    last_block = emoji_count_constraints = EmojiCountConstraintsBlock(
        inputs_path=doc_outlines.output_path,
        **pipeline_params,
        **mapper_params,
    )
    await emoji_count_constraints.run()
    feature_blocks.append(emoji_count_constraints)

    last_block = combine_constraints = CombineDatasetsBlock(
        inputs_paths=",".join(b.output_path for b in feature_blocks),
        **pipeline_params,
    )
    await combine_constraints.run()

    last_block = agg_group_constraints = ConstraintTypeAggregationBlock(
        inputs_path=last_block.output_path,
        group_field_names=["topic", "example_id", "document_template", "document_outline"],
        agg_field_names=["constraint_dct", "constraint_name", "constraint_type"],
        **pipeline_params,
    )
    await agg_group_constraints.run()

    if params.drafting_task.naturalize_constraints:
        last_block = ConstraintNaturalizationBlock(
            inputs_path=last_block.output_path,
            constraint_columns=["llm_judge_constraints", "verifiable_constraints"],
            model_name=params.drafting_task.constraint_naturalization_model_name,
            temperature=params.drafting_task.constraint_naturalization_temperature,
            max_tokens=params.drafting_task.constraint_naturalization_max_tokens,
            batch_size=params.drafting_task.constraint_naturalization_batch_size,
            **pipeline_params,
            **mapper_params,
        )
        await last_block.run()

    last_block = sample_non_verifialbe_constraints = ConstraintSamplingBlock(
        inputs_path=last_block.output_path,
        sample_per_header=params.llm_judge_constraint_sampling.non_verified_sample_per_header,
        max_sampled_constraints=params.llm_judge_constraint_sampling.non_verified_max_sampled_constraints,
        max_sampled_headers=params.llm_judge_constraint_sampling.non_verified_max_sampled_headers,
        constraints_type_col="llm_judge_constraints",
        exact_sampling=params.llm_judge_constraint_sampling.non_verified_exact_sampling,
        **pipeline_params,
        **mapper_params,
    )
    await sample_non_verifialbe_constraints.run()

    last_block = sample_verifialbe_constraints = ConstraintSamplingBlock(
        inputs_path=sample_non_verifialbe_constraints.output_path,
        sample_per_header=params.verifiable_constraint_sampling.verified_sample_per_header,
        max_sampled_constraints=params.verifiable_constraint_sampling.verified_max_sampled_constraints,
        max_sampled_headers=params.verifiable_constraint_sampling.verified_max_sampled_headers,
        constraints_type_col="verifiable_constraints",
        exact_sampling=params.verifiable_constraint_sampling.verified_exact_sampling,
        **pipeline_params,
        **mapper_params,
    )
    await sample_verifialbe_constraints.run()

    if params.verifiable_constraint_sampling.use_strict_header_names:
        last_block = header_name_constraints = HeaderNameConstraintsBlock(
            inputs_path=last_block.output_path,
            max_sampled_headers=0,  # for now, all headers
            **pipeline_params,
            **mapper_params,
        )
        await header_name_constraints.run()

    if params.drafting_task.verbalize_topic:
        last_block = PromptVariationsBlock(
            inputs_path=last_block.output_path,
            **pipeline_params,
            **mapper_params,
        )
        await last_block.run()

    # *************************
    # ****** Wrapping-up ******
    # *************************

    last_block = drafting_task = DraftingTaskBlock(
        inputs_path=last_block.output_path,
        use_strict_header_names=params.verifiable_constraint_sampling.use_strict_header_names,
        task_verbalizer_name=params.drafting_task.task_verbalizer_name,
        task_mode=params.drafting_task.task_mode,
        content_weight=params.drafting_task.content_weight,
        **pipeline_params,
        **mapper_params,
    )
    await drafting_task.run()

    # Stable, run-invariant size of the initial seed population (deduped wiki topics, capped at
    # max_examples), produced *before* any error-filtering downstream. Used as the fixed denominator
    # for a count-based, hash-keyed train/test split so a record's split never shifts between runs.
    seed_count = sum(1 for _ in iter_json_lines(seed_topics.output_path))

    return {
        "output": last_block.output_path,
        "seed_count": seed_count,
    }


async def drafting_synth_single_pipeline(params: Params) -> None:
    result_path = await drafting_synth_pipeline(params)
    logging.getLogger(params.pipeline_infra.pipeline_name).info("Completed pipeline")

    splits_block = TrainTestSplitBlock(
        inputs_path=result_path["output"],
        train_size=params.pipeline_infra.train_size,
        test_size=params.pipeline_infra.test_size,
        split_by_field=params.pipeline_infra.split_by_field or None,
        reference_size=result_path["seed_count"],
        output_dir=params.pipeline_infra.output_dir,
        pipeline_name=params.pipeline_infra.pipeline_name,
        log_level=params.pipeline_infra.log_level,
    )
    await splits_block.run()

    visualize_data(
        result_path["output"],
        logging.getLogger(params.pipeline_infra.pipeline_name),
        output_file=os.path.join(splits_block.task_output_dir, "data_summary.txt"),
    )

    # save pipeline CLI params
    dump_json(params.model_dump(), os.path.join(splits_block.task_output_dir, "pipeline_cli_params.json"))


if __name__ == "__main__":
    asyncio.run(drafting_synth_single_pipeline(parse_args()))
