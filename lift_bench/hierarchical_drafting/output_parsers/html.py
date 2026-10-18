"""Markdown -> HTML tree.

marko (GFM plus a generalized-list extension, see `markdown_lists.py`) renders the markdown to HTML, lxml parses it,
and the rules below map it onto HTML elements, clean it up and nest sections under their headings. The README's
"Section parser" section reports how often the result matches AI21's internal implementation on real model answers.

Behaviour worth knowing:
- Whitespace: line breaks inside text become spaces and <br> becomes a newline (see `_normalize_whitespace`).
- Tables under 2x2 are layout tables (read as paragraphs); larger ones are always data tables (see
  `_is_layout_table`).
- A heading with inline markup keeps its full text and its section body (see `_heading_title`), and `<<a b>>`
  stays literal text (see `_DOUBLE_ANGLE_BRACKETS`).

tests/.../fixtures/nested_html_expected_cases.jsonl freezes expected extractions on representative documents.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import lxml.html
from lxml.html import HtmlElement

from lift_bench.hierarchical_drafting.output_parsers.markdown_lists import (
    MARKER_STR_TO_LIST_TYPE,
    markdown,
)

ZERO_WIDTH_SPACE = "\u200b"


HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
TEXT_TAGS = ("text", "b", "i", "code")  # HTML_TextNode and its subclasses
_STYLED_TAGS = ("text", "b", "i", "code")  # isinstance(el, STYLED_ELEMENTS): Code subclasses TextNode
_ROOT_TAGS = (*HEADING_TAGS, "p", "data-table", "image", "code", "list", "page-break")
_COMPOSITE_TAGS = (*HEADING_TAGS, "p", "a")
_HTML_STYLE_TAGS = {"b": "b", "strong": "b", "i": "i"}


@dataclass
class Node:
    """An HTML element: `tag` is its XML tag, and per HTML convention it carries `text` or child `elements`."""

    tag: str
    text: Optional[str] = None
    elements: list["Node"] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# lxml tree -> HTML elements
# ---------------------------------------------------------------------------


def _concat_text_nodes(elements: list[Node]) -> list[Node]:
    result: list[Node] = []
    for el in elements:
        if el.tag in _STYLED_TAGS and not el.text:
            continue
        if result and el.tag in _STYLED_TAGS and el.tag == result[-1].tag:
            result[-1].text = (result[-1].text or "") + (el.text or "")
        else:
            result.append(el)
    return result


def _strip_surrounding_whitespaces(elements: list[Node]) -> list[Node]:
    for el in elements:
        if not _has_text_attr(el):
            break
        if not (el.text or "").strip():
            if el.text is None and el.elements:
                continue
            el.text = ""
            continue
        el.text = cast(str, el.text).lstrip()
        break
    for el in reversed(elements):
        if not _has_text_attr(el):
            break
        if not (el.text or "").strip():
            el.text = ""
            continue
        el.text = cast(str, el.text).rstrip()
        break
    return _concat_text_nodes(elements)


def _has_text_attr(el: Node) -> bool:
    """HTML types without a `text` field; the whitespace passes stop at them."""
    return el.tag not in ("list", "data-table", "image", "page-break")


def _code(element: HtmlElement, strip: bool = False) -> Node:
    text = element.text_content()
    lang = next(
        (c.replace("language-", "") for c in (element.get("class") or "").split(" ") if c.startswith("language-")),
        "",
    )
    return Node("code", text=text.strip() if strip else text, attrs={"lang": lang})


def _parse_pre(element: HtmlElement) -> list[Node]:
    code_children = element.xpath('.//code | .//div[@class="csharp"]')
    if code_children:
        return [Node("p", elements=[_code(c, strip=True) for c in code_children])]
    return [Node("p", text=element.text_content().strip(), attrs={"preformatted": True})]


def _parse_list(element: HtmlElement) -> list[Node]:
    marker = element.get("marker_type")
    list_type = MARKER_STR_TO_LIST_TYPE[marker] if marker else None
    items = [
        Node("litem", elements=_strip_surrounding_whitespaces(_parse_list_item_children(li)))
        for li in element
        if li.tag == "li"
    ]
    return [Node("list", elements=items, attrs={"type": list_type})]


def _parse_list_item_children(element: HtmlElement) -> list[Node]:
    match element.tag:
        case "code":
            return [_code(element)]
        case "a":
            text = element.text_content()
            return [Node("a", text=text)] if len(text) else []
        case "ul" | "ol" if not element.get("data-nested"):
            return _parse_list(element) if len(element.text_content()) else []
        case tag if tag in _HTML_STYLE_TAGS:
            return [Node(_HTML_STYLE_TAGS[tag], text=element.text_content())]
        case _:
            result = [Node("text", text=element.text)] if element.text else []
            for child in element:
                result.extend(_parse_list_item_children(child))
                if child.tail:
                    result.append(Node("text", text=child.tail))
            return result


def _extract_cell_text(cell: HtmlElement) -> str:
    text = unicodedata.normalize("NFKD", cell.text_content())
    text = re.sub(r"[ \t]+", " ", text)
    return text.replace("\\n", "\n").strip()


def _is_table_header(row: HtmlElement) -> bool:
    cells = row.xpath("./th | ./td")
    if not cells:
        return False
    if row.xpath("parent::thead") or all(cell.tag == "th" for cell in cells):
        return True
    if row.xpath('@class[contains(., "header")]') or row.xpath('@role="rowheader"'):
        return True
    non_empty = [c for c in cells if c.text_content().strip() or len(c.getchildren()) > 0]
    return bool(non_empty) and sum(c.tag == "th" for c in non_empty) / len(non_empty) > 0.5


def _is_layout_table(element: HtmlElement) -> bool:
    """Tables under 2x2 are layout tables (read as paragraphs); larger ones are data tables."""
    rows = element.xpath(".//tr")
    width = max((len(row.xpath(".//td|.//th")) for row in rows), default=0)
    return len(rows) < 2 or width < 2


def _parse_table(element: HtmlElement) -> list[Node]:
    """Rows as lists of (is_header, text) cells, split into head rows and body rows."""
    if not element.text_content() or _is_layout_table(element):
        return []
    head_rows, body_rows = [], []
    is_row_before_header = True
    for row in element.xpath("./tr | ./thead/tr | ./tbody/tr"):
        is_header_row = _is_table_header(row)
        cells = [
            (cell.tag == "th" or is_header_row, _extract_cell_text(cell) or ZERO_WIDTH_SPACE)
            for cell in row.xpath("./th | ./td")
        ]
        if is_row_before_header and is_header_row:
            head_rows.append(cells)
        else:
            body_rows.append(cells)
            is_row_before_header = False
    return [Node("data-table", attrs={"head": head_rows, "body": body_rows})]


def _parse_link(element: HtmlElement) -> list[Node]:
    text = element.text.strip() if element.text else None
    children = _parse_element_rec(element) if element.getchildren() else []
    if not text and not children:
        return []
    return [Node("a", text=text if text and not children else None, elements=children)]


def _try_parse_html_type(child: HtmlElement) -> list[Node]:
    nested = child.get("data-nested")
    match child.tag:
        case "table" if not nested:
            return _parse_table(child)
        case "a":
            return _parse_link(child)
        case "hr":
            return [Node("page-break")]
        case "pre":
            return _parse_pre(child)
        case "code":
            return [_code(child)]
        case "img":
            return [Node("image", attrs={"alternative": child.get("alt"), "source": child.get("src")})]
        case "ul" | "ol" if not nested:
            return _parse_list(child) if len(child.text_content()) else []
        case tag if tag in _HTML_STYLE_TAGS:
            nested_elements = _parse_element_rec(child)
            if any(el.tag in _ROOT_TAGS for el in nested_elements):
                return nested_elements
            return [Node(_HTML_STYLE_TAGS[tag], text=child.text_content())]
        case tag if tag in (*HEADING_TAGS, "p"):
            return [Node(tag, elements=_strip_surrounding_whitespaces(_parse_element_rec(child)))]
        case "td" | "th":
            return _parse_layout_table_cell(child)
    return []  # a container: the caller recurses into it


def _parse_layout_table_cell(cell: HtmlElement) -> list[Node]:
    """A cell of a layout table becomes its block elements, or a paragraph of its text."""
    if cell.findall(".//table"):
        return []
    children = _parse_element_rec(cell)
    root_elements = [el for el in children if el.tag in _ROOT_TAGS]
    return root_elements or [Node("p", elements=_strip_surrounding_whitespaces(children))]


def _parse_element_rec(element: HtmlElement) -> list[Node]:
    elements = [Node("text", text=element.text)] if element.text else []
    for child in element:
        if not isinstance(child.tag, str):  # comments and processing instructions
            if child.tail:
                elements.append(Node("text", text=child.tail))
            continue
        elements.extend(_try_parse_html_type(child) or _parse_element_rec(child))
        if child.tail:
            elements.append(Node("text", text=child.tail))
    return elements


def _handle_inline_root_elements(body_elements: list[Node]) -> list[Node]:
    """Wrap runs of non-root elements (stray text, styles) at the document root in a paragraph."""
    new_root_el: Optional[Node] = None
    result = []
    for el in body_elements:
        if el.tag not in _ROOT_TAGS:
            if new_root_el is None:
                new_root_el = Node("p")
            if new_root_el.elements or (el.text and el.text.strip()) or el.elements:
                new_root_el.elements.append(el)
        else:
            if new_root_el and new_root_el.elements:
                for curr in reversed(list(new_root_el.elements)):
                    if curr.text and curr.text.strip():
                        break
                    new_root_el.elements.pop()
                if new_root_el.elements:
                    new_root_el.elements[0].text = (new_root_el.elements[0].text or "").lstrip()
                    new_root_el.elements[-1].text = (new_root_el.elements[-1].text or "").rstrip()
            if new_root_el and new_root_el.elements:
                result.append(new_root_el)
            new_root_el = None
            result.append(el)
    return result


def _strip_verbose_text_tags(element: Node) -> None:
    if element.tag in (*_COMPOSITE_TAGS, "list", "litem"):
        if element.tag != "list" and not element.text:
            element.text = None
        if all(kid.tag == "text" for kid in element.elements):
            if element.tag != "list":
                element.text = (element.text or "") + "".join(kid.text for kid in element.elements if kid.text)
            element.elements = []
        curr_text_node: Optional[Node] = None
        filtered = []
        for kid in element.elements:
            if not curr_text_node and kid.tag in TEXT_TAGS:
                curr_text_node = kid
                filtered.append(kid)
            elif curr_text_node and kid.tag in TEXT_TAGS and kid.tag != curr_text_node.tag:
                curr_text_node = kid
                filtered.append(kid)
            elif curr_text_node and kid.tag == curr_text_node.tag:
                curr_text_node.text = (curr_text_node.text or "") + (kid.text or "")
            else:
                curr_text_node = None
                filtered.append(kid)
        element.elements = filtered
    for kid in element.elements:
        _strip_verbose_text_tags(kid)


def _cleanup_redundant_spaces(element: Node) -> None:
    if element.tag == "code" or (element.tag == "p" and element.attrs.get("preformatted")):
        return
    if element.text:
        element.text = re.sub(" +", " ", element.text)
    for kid in element.elements:
        _cleanup_redundant_spaces(kid)


# ---------------------------------------------------------------------------
# Section hierarchy
# ---------------------------------------------------------------------------

_IMPORTANCE = {**{h: i + 1 for i, h in enumerate(HEADING_TAGS)}, "p": 7, "data-table": 7, "list": 7}


def _deduce_hierarchy(node: Node) -> None:
    """Nest elements under the preceding, more important heading; paragraphs, tables and lists never parent."""
    parenting_stack: list[tuple[Node, int]] = []
    reparenting_pairs = []
    for element in node.elements:
        if element.tag in _COMPOSITE_TAGS:
            _deduce_hierarchy(element)
        if element.tag in _IMPORTANCE:
            importance = _IMPORTANCE[element.tag]
            while parenting_stack and importance <= parenting_stack[-1][1]:
                parenting_stack.pop()
            if parenting_stack:
                reparenting_pairs.append((element, parenting_stack[-1][0]))
            if importance < 7:
                parenting_stack.append((element, importance))
    for element, new_parent in reparenting_pairs:
        if new_parent.text:
            new_parent.elements.append(Node("text", text=new_parent.text, attrs={"is_title": True}))
            new_parent.text = None
        node.elements.remove(element)
        new_parent.elements.append(element)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_WRAPPING_MD_FENCE = ("```markdown\n", "\n```")
# Characters XML can't hold.
_XML_INVALID_CHARS = re.compile("[^\u0020-\ud7ff\u0009\u000a\u000d\ue000-\ufffd\U00010000-\U0010ffff]+")
# marko reads `<<a b>>` (IFBench's bigram wrapping) as an HTML tag and would drop it; it is kept literal.
_DOUBLE_ANGLE_BRACKETS = re.compile(r"<<([^<>\n]*)>>")


def _unwrap_markdown_fence(text: str) -> str:
    stripped = text.strip()
    prefix, suffix = _WRAPPING_MD_FENCE
    if stripped.startswith(prefix) and stripped.endswith(suffix):
        return stripped[len(prefix) : -len(suffix)]
    return text


def _normalize_whitespace(element: HtmlElement) -> None:
    """Line breaks inside text become spaces and <br> becomes a newline."""
    if element.tag in ("pre", "code"):
        return
    if element.text:
        element.text = element.text.replace("\n", " ")
    for child in element:
        if isinstance(child.tag, str):
            _normalize_whitespace(child)
        if child.tag == "br":
            child.tail = "\n" + (child.tail or "").lstrip()
        elif child.tail:
            child.tail = child.tail.replace("\n", " ")


def parse_markdown(text: str) -> Node:
    """The HTML body of a markdown document; an empty body when the document has no content."""
    text = _XML_INVALID_CHARS.sub("", text)
    if not text.strip():
        return Node("body")
    text = _DOUBLE_ANGLE_BRACKETS.sub(r"\<\<\1\>\>", _unwrap_markdown_fence(text))
    html = "<html><body>" + markdown(text) + "</body></html>"
    body = lxml.html.document_fromstring(html).find("body")
    _normalize_whitespace(body)
    elements = [el for el in _handle_inline_root_elements(_parse_element_rec(body)) if el.tag in _ROOT_TAGS]
    for el in elements:
        _strip_verbose_text_tags(el)
        _cleanup_redundant_spaces(el)
    for el in elements:
        if el.tag in HEADING_TAGS:
            el.attrs["title"] = _heading_title(el)
            for kid in el.elements:
                kid.attrs["is_title"] = True
    root = Node("body", elements=elements)
    _deduce_hierarchy(root)
    return root


def _heading_title(heading: Node) -> str:
    """A heading's full text, inline markup flattened."""
    if heading.text is not None:
        return heading.text
    return "".join(_flatten_text(el) for el in heading.elements)


def _flatten_text(node: Node) -> str:
    return (node.text or "") + "".join(_flatten_text(el) for el in node.elements)
