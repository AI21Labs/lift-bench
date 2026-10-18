"""nested_html extracts the expected text.

`fixtures/nested_html_expected_cases.jsonl` holds the expected extraction for selectors over the documents in
`fixtures/nested_html_expected_documents.jsonl`, covering how markdown maps to sections: paragraph separators,
line breaks, inline styles, loose / nested / task / lettered lists, code blocks as paragraphs, blockquotes, HTML,
images, tables and a wrapping markdown fence.
A case expecting None is one where the selected element doesn't exist.
"""

import json
from pathlib import Path

import pytest

from lift_bench.hierarchical_drafting.output_parsers.nested_html import (
    InvalidOutputFormatError,
    nested_html_parse,
)

FIXTURES = Path(__file__).parent / "fixtures"
DOCUMENTS = {
    d["document"]: d["markdown"]
    for d in map(json.loads, (FIXTURES / "nested_html_expected_documents.jsonl").read_text().splitlines())
}
CASES = [json.loads(line) for line in (FIXTURES / "nested_html_expected_cases.jsonl").read_text().splitlines()]


def _parse(markdown: str, selector_lst: list, is_shallow: bool = False) -> str:
    return nested_html_parse(markdown, selector_lst=selector_lst, is_shallow=is_shallow, document_type="markdown")


def _case_id(case: dict) -> str:
    path = "/".join(f"{tag}[{idx}]" for tag, idx in case["selector_lst"])
    return f"{case['document']}|{path}|{'shallow' if case['is_shallow'] else 'deep'}"


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_extraction_matches_the_expected_text(case):
    markdown = DOCUMENTS[case["document"]]

    if case["expected"] is None:
        with pytest.raises(InvalidOutputFormatError):
            _parse(markdown, case["selector_lst"], case["is_shallow"])
    else:
        assert _parse(markdown, case["selector_lst"], case["is_shallow"]) == case["expected"]


SECTION = [["h1", 0], ["h2", 0]]


def test_a_heading_with_inline_markup_keeps_its_full_text_and_its_body():
    markdown = "# T\n\n## **Intro** to `x`\n\nBody.\n"

    assert _parse(markdown, SECTION, is_shallow=True) == "Intro to x"
    assert _parse(markdown, SECTION) == "Body."


def test_double_angle_brackets_stay_literal():
    """Read as an HTML tag, `<<a b>>` would vanish, and IFBench's bigram-wrapping instruction could never pass."""
    assert _parse("# T\n\n## S\n\n<<new idea>> here.\n", SECTION) == "<<new idea>> here."


def test_tables_under_two_by_two_read_as_paragraphs():
    markdown = "# T\n\n## S\n\n| a |\n|---|\n| 1 |\n\n| b | c |\n|---|---|\n| 2 | 3 |\n"

    assert _parse(markdown, SECTION + [["p", 0]]) == "a"
    assert _parse(markdown, SECTION + [["data-table", 0]]) == "| b | c |\n| --- | --- |\n| 2 | 3 |"
