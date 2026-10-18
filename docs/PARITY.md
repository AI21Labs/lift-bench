# Parity with the internal implementation

This repository's section parser and scoring are independent implementations built on public libraries. They were
measured against the internal implementation, but they aren't guaranteed to return the same text or score on
every input, and scores computed here may differ from internal evaluations where the two disagree.

## Section parser

Each constraint is scored on the part of the answer it targets (a section, paragraph, list, table or code block),
extracted by the `nested_html` output parser in `lift_bench/hierarchical_drafting/output_parsers/`, built on marko
and lxml.

On 128 answers from two models (inkling-small, Claude Sonnet 4.6) to the 66 test tasks of a published split, it
returned the same text as the internal implementation for **100% of the 3,142 extractions the benchmark
performs** (3,675 evaluations), including 144 tables, 198 lists and 848 paragraphs. Known differences, none of which
affected those answers:

- **Not reproduced:** the internal implementation's boilerplate pruning in HTML extraction (it can drop raw-HTML
  blocks or content after a `---` rule), and its ML classifier for whether a table of 2×2 or more is a layout table
  (here such tables are always data tables).
- **Chosen differently:** a heading with inline markup keeps its full text and section body (the internal
  implementation reads it as empty), and `<<a b>>` stays literal text (the internal implementation reads it as an
  HTML tag).

`tests/lift_bench/hierarchical_drafting/fixtures/nested_html_expected_cases.jsonl` freezes the expected extractions
on representative documents; `test_nested_html_expected_extractions.py` checks them.

## Scoring

`tests/lift_bench/hierarchical_drafting/test_scoring_parity.py` scores 100 fixed tasks and completions (5 from each
published split's train and test files, 2,287 checks) and compares every check's score and each task's total with
the scores the internal implementation gave, stored in
`tests/lift_bench/hierarchical_drafting/fixtures/scoring_parity_cases.jsonl`. The judge's replies are recorded with
them and played back, so the test runs offline and fails if the critic's prompt changes.

It runs with the rest of the tests; run it alone after changing a metric, parser or the aggregation:

```bash
.venv/bin/python -m pytest tests/lift_bench/hierarchical_drafting/test_scoring_parity.py
```
