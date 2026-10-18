---
license: apache-2.0
pretty_name: LIFT-Bench
language:
  - en
task_categories:
  - text-generation
tags:
  - instruction-following
  - benchmark
  - document-drafting
  - verifiers
size_categories:
  - 10K<n<100K
configs:
  - config_name: short_doc_few_constraints
    data_files:
      - split: train
        path: short_doc_few_constraints/train.jsonl
      - split: test
        path: short_doc_few_constraints/test.jsonl
  - config_name: short_doc_some_constraints
    data_files:
      - split: train
        path: short_doc_some_constraints/train.jsonl
      - split: test
        path: short_doc_some_constraints/test.jsonl
  - config_name: short_doc_many_constraints
    data_files:
      - split: train
        path: short_doc_many_constraints/train.jsonl
      - split: test
        path: short_doc_many_constraints/test.jsonl
  - config_name: short_doc_numerous_constraints
    data_files:
      - split: train
        path: short_doc_numerous_constraints/train.jsonl
      - split: test
        path: short_doc_numerous_constraints/test.jsonl
  - config_name: long_doc_few_constraints
    data_files:
      - split: train
        path: long_doc_few_constraints/train.jsonl
      - split: test
        path: long_doc_few_constraints/test.jsonl
  - config_name: long_doc_some_constraints
    data_files:
      - split: train
        path: long_doc_some_constraints/train.jsonl
      - split: test
        path: long_doc_some_constraints/test.jsonl
  - config_name: long_doc_many_constraints
    data_files:
      - split: train
        path: long_doc_many_constraints/train.jsonl
      - split: test
        path: long_doc_many_constraints/test.jsonl
  - config_name: long_doc_numerous_constraints
    data_files:
      - split: train
        path: long_doc_numerous_constraints/train.jsonl
      - split: test
        path: long_doc_numerous_constraints/test.jsonl
  - config_name: flat_doc_some_constraints
    data_files:
      - split: train
        path: flat_doc_some_constraints/train.jsonl
      - split: test
        path: flat_doc_some_constraints/test.jsonl
  - config_name: flat_doc_many_constraints
    data_files:
      - split: train
        path: flat_doc_many_constraints/train.jsonl
      - split: test
        path: flat_doc_many_constraints/test.jsonl
---

# LIFT-Bench

**Localized Instruction-Following Tasks (LIFT-Bench)** is a benchmark of synthetic document-drafting tasks. A task
gives the model a topic and an outline, a nested tree of headers whose leaves hold paragraphs, code blocks, tables
and lists, and attaches constraints to individual elements of it.

Every constraint is *scoped*: it is scored only against the element it is attached to. That distinguishes "a
paragraph in French somewhere" from "section 2.1 in French", and makes constraint density a generation parameter
rather than an afterthought.

Each task mixes *verifiable* checks, run in code, with *judged* content requirements, run by an LLM judge, in a
single weighted score. A response that meets every formatting constraint but skips the content still fails every
judged constraint.

The scoring code, the `verifiers` taskset that runs it and the pipeline that synthesized the data are at
[github.com/AI21Labs/lift-bench](https://github.com/AI21Labs/lift-bench).

## Layout

The dataset has two axes. Each **config** is one of the ten generated datasets, and each config has a **train** and
a **test** split. The paper's **LIFT-Bench** is the `test` split of every config (784 tasks), and **LIFT-Train** is
the `train` split of every config (34,258 tasks).

```python
from datasets import load_dataset

bench = load_dataset("ai21labs/LIFTBench", "short_doc_few_constraints", split="test")
train = load_dataset("ai21labs/LIFTBench", "short_doc_few_constraints", split="train")
```

The configs come from three generation runs: short outlines, long outlines and flat (no outline). Within a run every
config shares the same topics and outlines; only the number of constraints attached to them changes.

| Config | `dataset_id` | Train | Test |
| --- | --- | --- | --- |
| `short_doc_few_constraints` | `l3_d2_sh3_sv2_snv2` | 3,681 | 85 |
| `short_doc_some_constraints` | `l3_d2_sh5_sv3_snv2` | 3,681 | 85 |
| `short_doc_many_constraints` | `l3_d2_sh7_sv3_snv3` | 3,681 | 85 |
| `short_doc_numerous_constraints` | `l3_d2_sh7_sv4_snv4` | 3,681 | 85 |
| `long_doc_few_constraints` | `l5_d2_sh5_sv2_snv2` | 2,978 | 66 |
| `long_doc_some_constraints` | `l5_d2_sh7_sv3_snv2` | 2,978 | 66 |
| `long_doc_many_constraints` | `l5_d2_sh7_sv3_snv3` | 2,978 | 66 |
| `long_doc_numerous_constraints` | `l5_d2_sh7_sv4_snv4` | 2,978 | 66 |
| `flat_doc_some_constraints` | `flat_sv3_snv3` | 3,811 | 90 |
| `flat_doc_many_constraints` | `flat_sv5_snv5` | 3,811 | 90 |

`dataset_id` encodes the generation parameters:

| Code | Meaning |
| --- | --- |
| `l<n>`, `d<n>` | Outline shape: up to `n` children per header, `n` levels deep |
| `sh<n>` | Up to `n` sections carry constraints |
| `sv<n>`, `snv<n>` | Up to `n` verifiable and `n` judged constraints per section (per task, for flat) |
| `flat` | No outline; every constraint applies to the whole response |

## Columns

| Column | Type | Content |
| --- | --- | --- |
| `prompt` | list of messages | The task shown to the model, one user message |
| `info` | dict | `{"evaluations": [...]}`, the rubric the task is scored against |
| `example_id` | string | Identifier of the task |
| `topic` | string | Seed topic, a Wikipedia article title |
| `document_outline` | string | The markdown outline the task was built from |
| `dataset_name` | string | The config name |
| `dataset_id` | string | The generation-parameter code |
| `split` | string | `train` or `test` |
| `metadata` | dict | Empty |

Each entry of `info.evaluations` is one constraint:

```json
{
  "evaluation_id": "content_verifiable_constraints_3",
  "weight_group": "content",
  "weight": 0.175,
  "metric_name": "lexical_count",
  "metric_args": { "string": "climate", "lower_bound": 2, "upper_bound": 5 },
  "parser_name": "nested_html",
  "parser_args": { "selector_lst": [["H2", 1]], "is_shallow": false, "document_type": "markdown" }
}
```

`parser_name` and `parser_args` name the section of the document the constraint is scored on; `metric_name` and
`metric_args` name the check. The score of a task is the weighted sum of its per-constraint scores.

## Scoring

Score a model with the `drafting_constraint` taskset from the repository, which runs under
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers):

```bash
hf download ai21labs/LIFTBench --repo-type dataset --local-dir data/splits
ENV_ID=drafting_constraint VLLM_MODEL=<model-name> ./run_eval.sh \
    --env.taskset.dataset-path data/splits/short_doc_few_constraints/test.jsonl
```

Judged constraints are scored by the LLM judge each task names in `metric_args.critic_model_name`,
`google/gemma-4-31B-it`, called through an OpenAI-compatible endpoint. `--env.taskset.judge-model <model-name>`
scores them with another judge instead, without changing the data. Results with a different judge are not
comparable to the published ones.

## Loading notes

- The rubric fields `metric_args` and `parser_args.selector_lst` hold values whose type varies between rows, which
  `datasets` stores as JSON fields and decodes on access. Loading was verified with `datasets` 4.8 and 5.1.
- When `datasets` loads the files it re-encodes them, and constraint `weight`s come back rounded to 10 decimal
  places (0.16666666666666666 becomes 0.1666666667). A task's score changes by less than 1e-9 as a result. The JSONL
  files themselves hold the exact values, and the `verifiers` taskset reads them directly.

## License and attribution

The data is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). Topics are titles
of Wikipedia's vital articles, available under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

<!-- TODO: add the paper's BibTeX once it is public. -->
