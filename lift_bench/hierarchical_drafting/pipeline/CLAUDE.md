# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All `invoke` commands are run from the **repo root**:

```bash
# Tests
uv run pytest                              # run all tests
uv run pytest tests/path/to/test.py        # run a single test file
invoke test --coverage                     # run with coverage report

# Linting / formatting
invoke formatter                           # check ruff formatting
invoke formatter --fix                     # auto-fix formatting
invoke lint                                # ruff lint check
invoke lint --fix                          # auto-fix lint issues

# Run the pipeline
python -m lift_bench.hierarchical_drafting.pipeline.main --pipeline_infra.max_examples 100 --pipeline_infra.output_dir ./output
```

The pipeline needs an OpenAI-compatible model endpoint, set by `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## Architecture

This is an **async block pipeline** that generates hierarchical drafting tasks for alignment training data. The pipeline chains `AtomicBlock` subclasses sequentially, each writing JSONL output consumed by the next.

### Pipeline stages (`main.py`)

```text
Wiki topics → Document templates → Document outlines
    → [Parallel constraint generation] → Combine → Aggregate
    → Sample LLM-as-a-judge constraints → Sample verifiable constraints
    → (Optional) Header name constraints
    → DraftingTaskBlock → TrainTestSplitBlock
```

### Two constraint families

**LLM-as-a-judge (non-verifiable)** — an LLM judge evaluates quality after generation:

- `ContentConstraintsBlock`: content that should appear in a section
- `CodeblockConstraintsBlock`, `TableConstraintsBlock`, `ListConstraintsBlock`: artifact suggestions

**Verifiable** — automatically checkable rules applied to the final document:

- `TargetLanguageConstraintsBlock`, `CaseTypeConstraintsBlock`, `LexicalCountConstraintsBlock`, `EmojiCountConstraintsBlock`, `HeaderNameConstraintsBlock`

All constraint blocks live under `blocks/llm_judge/` or `blocks/verifiable/`. Both families produce `VerifiableConstraint` objects (from `constraints/`) with different `constraint_type` values (`"llm_judge_constraints"` vs `"verifiable_constraints"`).

### Block pattern

Each block inherits from `AtomicBlock`, `MapperBlock`, or `BatchMapperBlock`.
Blocks accept `inputs_path` + shared params (`output_dir`, `pipeline_name`, `log_level`, `max_concurrency`, `chunk_size`).
Every block writes its output to a JSONL file at `output_path`.

### Key supporting modules

- `cli.py` — all CLI arguments and defaults as the pydantic-settings `Params` model (YAML `--config`, CLI overrides)
- `utils.py` — `merge_dicts_of_lists`, flatten/restore helpers, `sample_dict_across_sublists`
- `analytics.py` — `visualize_data()` logs post-run statistics (depth, leaf count, constraint distribution)
- `blocks/constraints_aggregation_block.py` — groups constraint records by `["topic", "example_id", "document_template", "document_outline"]` before sampling
- `blocks/constraint_sampling_block.py` — samples LLM-as-a-judge and verifiable constraints independently (called twice in `main.py`)

### Data representation

Documents are hierarchical trees (`DocumentNode`) with HTML tags (H1–H3, P, CODE, DATA_TABLE, LIST). The `DraftingTaskBlock` converts these into LLM prompt/message format with evaluation specs and an `aggregation_config` that balances structure vs. content scoring.
