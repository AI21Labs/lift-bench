# Pipeline Architecture Reference

Precise notes on the task synthesis pipeline for LIFT-Bench.
Intended as a reference for generating diagrams, documentation, and future development.

## Pipeline stages

Defined in `pipeline/main.py: drafting_synth_pipeline`. Each stage is an `AtomicBlock`
subclass that reads one JSONL file and writes another.

### Sequential stages

| Stage | Class | Key params | Output |
| --- | --- | --- | --- |
| 1 | `GenerateSeedDraftingConfigBlock` | `seed_method="wiki_topics"` | seed config |
| 2 | `SeedTopicsBlock` | `max_items` | wiki topic list |
| 3 | `DocumentOutlineTemplateBlock` | `max_children`, `max_depth`, `template_verbalizer_name` | structural skeletons |
| 4 | `DocumentOutlineBlock` | `model_name`, `batch_size`, `max_tokens`, `temperature` | LLM-expanded outlines |
| 5 | (parallel — see below) | | constraint candidates |
| 6 | `CombineDatasetsBlock` | `inputs_paths` (comma-joined) | merged constraints |
| 7 | `ConstraintTypeAggregationBlock` | `group_field_names`, `agg_field_names` | grouped by example |
| 7b | `ConstraintNaturalizationBlock` | optional (`naturalize_constraints`), `batch_size` | LLM-rephrased `custom_text` per constraint |
| 8 | `ConstraintSamplingBlock` | `constraints_type_col="llm_judge_constraints"` | sampled LLM-as-a-judge constraints |
| 9 | `ConstraintSamplingBlock` | `constraints_type_col="verifiable_constraints"` | sampled verifiable constraints |
| 10 | `HeaderNameConstraintsBlock` | optional (`use_strict_header_names`) | header lexical constraints |
| 10b | `PromptVariationsBlock` | optional (`verbalize_topic`) | `prompt_variations` (LLM rewrites of the topic opening) |
| 11 | `DraftingTaskBlock` | `task_verbalizer_name`, `use_strict_header_names` | `prompt` + `info` |
| 12 | `TrainTestSplitBlock` | `train_size`, `test_size` | split JSONL files |

### Parallel constraint generation (stage 5)

All constraint blocks read from the same document outlines output. They run independently and are merged in stage 6.

**LLM-as-a-judge (non-verifiable)** — LLM judge evaluates after generation:

| Block | Constraint name | Artifact tag | Guard |
| --- | --- | --- | --- |
| `ContentConstraintsBlock` | `content` | none (leaf P) | `max_content_constraints > 0` |
| `CodeblockConstraintsBlock` | `code` | `HTMLTag.CODE` | `max_artifact_constraints > 0` |
| `TableConstraintsBlock` | `table` | `HTMLTag.DATA_TABLE` | `max_artifact_constraints > 0` |
| `ListConstraintsBlock` | `list` | `HTMLTag.LIST` | `max_artifact_constraints > 0` |

**Verifiable** — deterministic metric checks:

| Block | Constraint name | What it checks |
| --- | --- | --- |
| `TargetLanguageConstraintsBlock` | `language` | `langid` language detection |
| `CaseTypeConstraintsBlock` | `case_type` | upper / lower / title casing |
| `LexicalCountConstraintsBlock` | `lexical_count` | keyword frequency + presence/absence |
| `EmojiCountConstraintsBlock` | `emoji_count` | emoji count in range |
| `HeaderNameConstraintsBlock` | `doc_structure` | exact header name match (check_title=True) |
| `IFBenchConstraintsBlock` | `ifbench` | IFTrain instructions; opt-in via `--ifbench_constraint_sampling.use_ifbench_constraints` |

`doc_structure` constraints get `weight_group = "structure"` (10% budget). All others get `weight_group = "content"` (90% budget), configurable via `--drafting_task.content_weight`.

## Document representation

`DocumentNode` is a recursive tree node (`document/document_node.py`).

```text
DocumentNode
  type: HTMLTag | None        # None only on root
  value: str                 # header text, paragraph content, etc.
  children: List[DocumentNode]
  parent: Optional[DocumentNode]
```

**`HTMLTag` values** (structural): `H1 H2 H3 H4 H5 H6`
**`HTMLTag` values** (content): `P CODE DATA_TABLE LIST IMAGE B I TEXT_NODE LIST_ITEM META BODY`

Key methods:

- `get_selector()` → XPath-like list `[type1, idx1, type2, idx2, ...]` used by `nested_html` parser
- `find_node_by_value(name)` → locate header by exact name
- `is_leaf()` → no children
- `from_md(text)` / `to_md()` → serialise to/from markdown headers

## Constraint data model

### `VerifiableConstraintMetadata` (dataclass_json)

```python
constraint:           VerifiableConstraint  # the actual constraint object
artifact_html_tag:     Optional[HTMLTag]      # sub-element scope (CODE, DATA_TABLE, LIST)
artifact_html_tag_idx: Optional[int]         # specific instance; None = auto-increment
check_title:          bool                  # True = check header text only
```

### `VerifiableConstraint.to_evaluation_dict()`

Returns a dict that becomes one entry in `info["evaluations"]`. Shape differs by constraint family:

**Verifiable metrics** (deterministic):

```python
{
  "metric_name": "emoji_count" | "lexical_match" | "lexical_count" | "case_type" | "target_language",
  "metric_args": { ... }   # metric-specific
}
```

**LLM-as-a-judge critics** use the same shape; the judge settings, criterion and request travel in `metric_args`:

```python
{
  "metric_name": "light_absolute_critic" | "parser_evaluator",
  "metric_args": {
    "critic_model_name": ..., "n": ..., "temperature": ..., "max_tokens": ..., "use_binary_range": False,
    "criteria": [...], "messages": [...]   # light_absolute_critic
    # or "should_succeed": True            # parser_evaluator
  }
}
```

`DraftingTaskBlock._evaluators_for_task` then adds:

- `parser_name = "nested_html"` + `parser_args` (selector path, is_shallow, document_type)
- `weight_group = "structure" | "content"`
- `evaluation_id = "structure_{type}_{idx}" | "content_{type}_{idx}"`

## Constraint sampling

`ConstraintSamplingBlock` — called twice, once per constraint family.

```text
sample_per_header:          bool  # sample each header independently (default True)
max_sampled_constraints:    int   # per header: 0=skip, -1=all, N=sample N
max_sampled_headers:        int   # 0=all headers, N=sample N leaf headers
exact_sampling:             bool  # True=exactly N, False=between 1 and N
constraints_type_col:       str   # "llm_judge_constraints" | "verifiable_constraints"
```

Output key: `item[f"sampled_{constraints_type_col}"]`

## Weight allocation

`assign_weights` in `pipeline/blocks/drafting_task.py` — applied by `DraftingTaskBlock` after sampling, before writing output.

```text
structure evaluations  →  weight = (1 - content_weight) / len(structure_evals)  each
content evaluations    →  weight = content_weight / len(content_evals)          each
```

With no structure evaluations (flat tasks), content takes the full weight. A task with no content evaluations is
written to the block's error file instead of the output.

## Task verbalization

Registered via `@register_task_verbalizer` in `pipeline/task_verbalization/`.

| Name | Class | Prompt style |
| --- | --- | --- |
| `trailing_constraints` | `DefaultTaskVerbalizer` | Outline first, constraints appended after |
| `inline_constraints` | `InlineConstraintsTaskVerbalizer` | Constraints embedded next to each header |
| `flat` | `FlatTaskVerbalizer` | Opening plus a bullet list of constraints, no outline |
| `ifeval_flat` | `IFEvalFlatTaskVerbalizer` | Constraints woven into prose; artifact groups as numbered blocks |

`DraftingTaskBlock` picks randomly when `task_verbalizer_name="random"`, else uses the named one.
When the item has `prompt_variations`, one is chosen per topic (md5-seeded) and replaces the templated opening.

Constraint text comes from `task_verbalization/utils.render_constraint_text`: the metadata's `custom_text`
(set by `ConstraintNaturalizationBlock`) when present, else `get_constraint_verbalizer(constraint)` →
`verbalizer.verbalize(constraint).to_prompt()`.

## Output parser: `nested_html`

Registered in `output_parsers/registry.py`. Scopes metric evaluation to a specific section.

```text
selector_lst:  List[Tuple[str, int]]   # e.g. [["H1", 0], ["H2", 1], ["CODE", 0]]
is_shallow:    bool                    # True = heading text only; False = everything below the heading
document_type: "markdown"
fail_silently: bool                    # default True → return "" on parse error
```

Parser walks the markdown tree following `(tag, index)` pairs (0-based, negative indices allowed).
Deep extraction (`is_shallow=False`) of a header drops that header's own heading line, which the prompt dictates,
but keeps sub-section headings; non-header nodes (paragraph, list, table, code) render whole.
All metrics are wrapped with `with_output_parser()` so the parser runs transparently before metric evaluation.

## Metrics registry

All metrics registered via `@register_metric(name)` in `metrics/`. A metric returns its verdict (1.0 / 0.0, or a ratio);
`with_output_parser` maps it to `VALID_PARSING_SCORE + (1 - VALID_PARSING_SCORE) * verdict` (0.00390625 for a failed
check), and to 0.0 when the target is missing from the answer or the metric raises
`ParsingError` on input it can't parse. Errors (`metrics/errors.py`):

- Arguments are validated against the metric's signature (required, typed with lax coercion, no extra keys).
- A check that can't run (invalid arguments, `MetricError`, any other exception) raises
  `EvaluationError`. `constraint_reward` still runs every check, then raises `EvaluationError` for the task, which has
  no score; the `verifiers` tasksets record it as the trace's error and leave its reward unscored (0.0).

| `metric_name` | Module | Key `metric_args` | Returns |
| --- | --- | --- | --- |
| `emoji_count` | `emoji_count.py` | `lower_bound`, `upper_bound` | 1.0 if count in range |
| `lexical_match` | `lexical_match.py` | `string`, `match_type`, `should_match`, `fuzzy_match` | 1.0 if match == should_match |
| `lexical_count` | `lexical_count.py` | `string`, `lower_bound`, `upper_bound`, `fuzzy_match`, `strict_word_boundaries` | 1.0 if count in range |
| `case_type` | `case_type.py` | `case_type` (`upper`/`lower`/`title`) | 1.0 if all words conform |
| `target_language` | `target_language.py` | `language_code` | 1.0 if langid matches |
| `ifbench` | `ifbench.py` | `instruction_id_list`, `kwargs` | 1.0 if every instruction is followed |
| `light_absolute_critic` | `critic.py` | `critic_model_name`, `criteria`, `messages`, `use_binary_range`, `n`, `temperature`, `max_tokens` | 1.0 if the LLM judge's score reaches `good_completion_threshold` (0.7) |
| `parser_evaluator` | `parser_evaluator.py` | `should_succeed` | `should_succeed` if the scoped section/artifact parses, else 0.0 |

`constraint_reward(completion, info)` returns:

```python
{
  "score":           float,              # weighted sum
  "agg_eval_scores": list[float],        # raw per-constraint scores
  "agg_eval_ids":    list[str] | None    # evaluation_id per constraint (None if absent)
}
```

## Final output schema

Each JSONL row written by `DraftingTaskBlock`:

```json
{
  "prompt": [{"role": "user", "content": "Write a markdown document..."}],
  "info": {
    "evaluations": [
      {
        "evaluation_id":  "structure_doc_structure_0",
        "weight_group":   "structure",
        "weight":         0.15,
        "metric_name":    "lexical_match",
        "metric_args":    {"string": "Header Name", "match_type": "exact", "should_match": true},
        "parser_name":    "nested_html",
        "parser_args":    {"selector_lst": [["H1", 0]], "is_shallow": true, "document_type": "markdown"}
      },
      {
        "evaluation_id":  "content_verifiable_constraints_1",
        "weight_group":   "content",
        "weight":         0.175,
        "metric_name":    "target_language",
        "metric_args":    {"language_code": "fr"},
        "parser_name":    "nested_html",
        "parser_args":    {"selector_lst": [["H1", 0], ["H2", 1]], "is_shallow": false, "document_type": "markdown"}
      }
    ]
  },
  "metadata":                    {},
  "topic":                       "Renewable Energy",
  "document_outline":            "# Renewable Energy\n## Solar Power\n...",
  "document_template":           "...",
  "example_id":                  "abc123",
  "sampled_llm_judge_constraints":  {...},
  "sampled_verifiable_constraints": {...}
}
```

## IFTrain / IFBench integration

Module: `hierarchical_drafting/iftrain/`

- `instructions.py` — 60+ checker classes (KeywordChecker, ParagraphChecker, etc.)
- `instructions_registry.py` — `FUNCTION_DICT` + `INSTRUCTION_DICT` map names → checker classes
- Instruction categories: keywords, language, length, content, format, combination

**Status: opt-in.** `--ifbench_constraint_sampling.use_ifbench_constraints` adds `IFBenchConstraintsBlock`, which samples
`randint(1, --ifbench_constraint_sampling.ifbench_max_sampled_constraints)` non-conflicting instructions per leaf header from
`--ifbench_constraint_sampling.ifbench_custom_instructions_lst` (default: 19 instructions, listed in `pipeline/cli.py`). Instruction kwargs are the
`build_description()` defaults.
