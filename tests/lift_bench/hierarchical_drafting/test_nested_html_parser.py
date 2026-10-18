"""Scope of the nested_html parser.

For a header, the two extraction modes partition the section: is_shallow=True
selects the heading text, is_shallow=False selects everything below it. The
heading itself belongs to neither side of a content constraint -- the prompt
dictates it verbatim, so scoring it would judge the model on text it was
ordered to write.

Nodes that are not headers have no heading line to drop, and their content
lives in `node.value` (or, for a list, in items that need their bullets), so
deep extraction must leave them untouched.
"""

import pytest

from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.metrics.case_type import (
    CaseType,
    _inaccurate_words,
)
from lift_bench.hierarchical_drafting.output_parsers.nested_html import (
    nested_html_parse,
)


DOC = """# Renewable Energy

Opening paragraph.

## Solar Power

solar power is captured by photovoltaic cells.

- cheap
- scalable

| Cell | Efficiency |
| --- | --- |
| mono | 22% |

```python
print("hello")
```

## Wind Power

WIND POWER USES TURBINES.
"""

SOLAR = [(HTMLTag.H1, 0), (HTMLTag.H2, 0)]


def _parse(selector, is_shallow=False):
    return nested_html_parse(DOC, selector_lst=selector, is_shallow=is_shallow, document_type="markdown")


def test_section_body_excludes_its_own_heading():
    body = _parse(SOLAR)
    assert not body.startswith("#")
    assert "Solar Power" not in body
    assert body.startswith("solar power is captured")


def test_shallow_extraction_returns_only_the_heading_text():
    assert _parse(SOLAR, is_shallow=True) == "Solar Power"


def test_section_body_keeps_every_kind_of_child_content():
    body = _parse(SOLAR)
    assert "- cheap" in body
    assert "| mono | 22% |" in body
    assert 'print("hello")' in body


def test_sibling_sections_stay_out_of_scope():
    body = _parse(SOLAR)
    assert "WIND POWER" not in body


def test_lowercase_section_body_is_satisfiable_under_a_title_case_heading():
    """A `lower` constraint on a section whose heading the prompt fixes as Title
    Case must be satisfiable, so the heading is not scored as part of the body."""
    doc = "# Solar Power\n\nsolar power is captured by photovoltaic cells.\n"
    body = nested_html_parse(doc, selector_lst=[(HTMLTag.H1, 0)], is_shallow=False, document_type="markdown")
    assert _inaccurate_words(body, CaseType.LOWER) == []


def test_section_with_no_content_has_an_empty_body():
    body = nested_html_parse("# Solar Power\n", selector_lst=[(HTMLTag.H1, 0)], document_type="markdown")
    assert body == ""


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ([(HTMLTag.H1, 0), (HTMLTag.H2, 0), (HTMLTag.DATA_TABLE, 0)], "| mono | 22% |"),
        ([(HTMLTag.H1, 0), (HTMLTag.H2, 0), (HTMLTag.LIST, 0)], "- cheap"),
        ([(HTMLTag.H1, 0), (HTMLTag.H2, 0), (HTMLTag.CODE, 0)], 'print("hello")'),
        ([(HTMLTag.H1, 0), (HTMLTag.P, 0)], "Opening paragraph."),
    ],
)
def test_non_header_nodes_render_whole(selector, expected):
    """These carry their content in `node.value`, not in children -- rendering
    only their children would return an empty string and silently zero out
    every table, list, code and paragraph constraint."""
    assert expected in _parse(selector)


def test_parent_section_body_still_contains_its_subsections():
    """Constraints only ever land on leaf sections (see ConstraintSamplingBlock),
    but a parent's body keeps its subsections rather than losing them."""
    body = _parse([(HTMLTag.H1, 0)])
    assert "Renewable Energy" not in body
    assert "Opening paragraph." in body
    assert "## Solar Power" in body
