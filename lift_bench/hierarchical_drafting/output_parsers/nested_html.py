"""
Markdown document parser with XPath-style element selection.

Parses a markdown completion into an HTML tree (see `html.py`) and extracts the element a selector path of
(HTMLTag, index) pairs points at (e.g. H1[0] -> H2[1] -> CODE[0]):
shallow extraction returns the element's own text, deep extraction renders it back to markdown.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.document.types import DocumentTextType
from lift_bench.hierarchical_drafting.output_parsers.html import (
    HEADING_TAGS,
    Node,
    parse_markdown,
)
from lift_bench.hierarchical_drafting.output_parsers.html_export import (
    to_markdown,
)


class ParsingError(Exception):
    pass


class InvalidOutputFormatError(ParsingError):
    pass


def _xpath_steps(tag: HTMLTag) -> list[str]:
    """Code blocks live inside paragraphs in HTML, so CODE selects `p/code`."""
    return ["p", "code"] if tag == HTMLTag.CODE else [tag.value]


def _xpath_str(selector_lst: List[Tuple[HTMLTag, int]]) -> str:
    def index(i: int) -> str:
        return str(i + 1) if i >= 0 else f"last() - {-1 - i}"

    prefix = "".join(f"{'/'.join(_xpath_steps(t))}[{index(i)}]/" for t, i in selector_lst[:-1])
    last_tag, last_index = selector_lst[-1]
    return f"({prefix}{'/'.join(_xpath_steps(last_tag))})[{index(last_index)}]"


def _pick(matches: list[Node], idx: int) -> list[Node]:
    position = idx if idx >= 0 else len(matches) + idx
    return [matches[position]] if 0 <= position < len(matches) else []


def _children(nodes: list[Node], steps: list[str]) -> list[Node]:
    for tag in steps:
        nodes = [child for node in nodes for child in node.elements if child.tag == tag]
    return nodes


def _select(body: Node, selector_lst: List[Tuple[HTMLTag, int]]) -> Node:
    """Every step but the last indexes within its parent; the last indexes across all matches."""
    context = [body]
    for tag, idx in selector_lst[:-1]:
        steps = _xpath_steps(tag)
        context = [
            picked for node in _children(context, steps[:-1]) for picked in _pick(_children([node], steps[-1:]), idx)
        ]
    last_tag, last_index = selector_lst[-1]
    selected = _pick(_children(context, _xpath_steps(last_tag)), last_index)
    if not selected:
        raise InvalidOutputFormatError(f"xpath '{_xpath_str(selector_lst)}' doesn't exist.")
    return selected[0]


class NestedHTMLParserConfig(BaseModel):
    html_tag: Optional[HTMLTag] = None
    index: Optional[int] = None
    selector_lst: List[Tuple[HTMLTag, int]] = Field(default_factory=list)
    document_type: DocumentTextType = DocumentTextType.MARKDOWN
    # For a header, the two modes partition the section: True selects its heading
    # text, False selects everything below that heading. A section's heading is
    # dictated by the prompt ("headers should *exactly* match the outline"), so
    # leaving it inside the body would score constraints against text the model
    # was ordered to write, e.g. a lowercase constraint on a Title Case section
    # could never pass.
    is_shallow: bool = False
    fail_silently: bool = True

    def model_post_init(self, __context) -> None:
        has_tag = self.html_tag is not None
        has_idx = self.index is not None
        assert has_tag == has_idx, "Must provide both html_tag and index, or neither"
        assert (len(self.selector_lst) == 0) != (self.html_tag is None), (
            "Provide either selector_lst or html_tag+index, not both (and not neither)"
        )
        if has_tag and has_idx:
            self.selector_lst = [(self.html_tag, self.index)]


class NestedHTMLParser:
    """Extracts a specific element from a markdown document by selector path."""

    def __init__(self, config: Optional[NestedHTMLParserConfig] = None):
        if config is None:
            raise ValueError("config is required")
        self.config = config

    def parse_output(self, output: str) -> str:
        if self.config.document_type != DocumentTextType.MARKDOWN:
            raise ParsingError(
                f"Unsupported document type: {self.config.document_type!r}. Only 'markdown' is supported."
            )
        try:
            body = parse_markdown(output)
        except Exception as exc:
            if not self.config.fail_silently:
                raise ParsingError(f"Failed to parse markdown. Text:\n{output}") from exc
            body = Node("body")

        try:
            node = _select(body, self.config.selector_lst)
        except InvalidOutputFormatError:
            raise
        except Exception as exc:
            raise ParsingError(f"Failed to select {_xpath_str(self.config.selector_lst)!r}. Text:\n{output}") from exc
        if self.config.is_shallow:
            return self._extract_shallow(node)
        return self._extract_deep(node)

    @staticmethod
    def _extract_deep(node: Node) -> str:
        """A heading's section without its own heading line; any other element whole."""
        if node.tag not in HEADING_TAGS:
            return to_markdown(Node("body", elements=[node]))
        return to_markdown(Node("body", elements=[el for el in node.elements if not el.attrs.get("is_title")]))

    @staticmethod
    def _extract_shallow(node: Node) -> str:
        if node.tag == "data-table":
            return to_markdown(Node("body", elements=[node])).splitlines()[0] if node.attrs["head"] else ""
        if node.tag in HEADING_TAGS:
            # The full heading text, so a heading that starts with bold, a link or code doesn't read as "".
            return node.attrs["title"].rstrip()
        first = node.elements[0] if node.elements else None
        text = node.text or (first.text if first is not None and first.tag == "text" else "")
        return (text or "").rstrip()


from lift_bench.hierarchical_drafting.output_parsers.registry import (  # noqa: E402
    register_parser,
)


@register_parser("nested_html")
def nested_html_parse(text: str, **parser_args) -> str:
    """Raises ParsingError when the parser can't be configured or the answer can't be parsed or searched. An error
    while rendering the selected element propagates unchanged."""
    try:
        config = NestedHTMLParserConfig(**parser_args)
    except Exception as exc:
        raise ParsingError(f"Failed to create output parser nested_html: {exc}") from exc
    return NestedHTMLParser(config).parse_output(text)
