"""HTML tree -> markdown for the element types markdown parsing can produce, with styles and headings kept and
images, links and HTML tables left out."""

from collections import deque
from string import ascii_lowercase, ascii_uppercase

from lift_bench.hierarchical_drafting.output_parsers.html import (
    HEADING_TAGS,
    ZERO_WIDTH_SPACE,
    Node,
)
from lift_bench.hierarchical_drafting.output_parsers.markdown_lists import (
    LIST_TYPE_TO_MARKER_STR,
    ListType,
)

_ROMAN_VALUES = [(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _number_to_roman(number: int) -> str:
    result = ""
    for value, letters in _ROMAN_VALUES:
        while number >= value:
            result += letters
            number -= value
    return result


def _list_prefix(list_type: ListType, idx: int) -> str:
    match list_type:
        case ListType.ORDERED_LOWERCASE_ROMAN_NUMERALS:
            return f"{_number_to_roman(idx + 1).lower()}."
        case ListType.ORDERED_UPPERCASE_ROMAN_NUMERALS:
            return f"{_number_to_roman(idx + 1)}."
        case ListType.ORDERED_LOWERCASE_LETTERS:
            return f"{ascii_lowercase[idx]}."
        case ListType.ORDERED_UPPERCASE_LETTERS:
            return f"{ascii_uppercase[idx]}."
        case ListType.ORDERED_DECIMALS:
            return f"{idx + 1}."
        case _:
            return LIST_TYPE_TO_MARKER_STR[list_type]


def _cell(is_header: bool, text: str) -> str:
    content = text.replace("\n", " ").replace(ZERO_WIDTH_SPACE, "")
    return f"**{content}** " if is_header else content + " "


def _table_to_markdown(table: Node) -> str:
    head_rows, body_rows = table.attrs["head"], table.attrs["body"]
    md = ""
    if head_rows:
        divider = ""
        for _, text in head_rows[0]:
            md += f"| {text.replace(chr(10), ' ')} "
            divider += "| --- "
        md += "|\n" + divider + "|\n"
        for row in head_rows[1:]:
            md += "".join(f"| {_cell(is_header, text)}" for is_header, text in row) + "|\n"
    else:
        num_cells = len(body_rows[0])
        md += "".join(["| "] * num_cells + ["|\n"] + ["| --- "] * num_cells + ["|\n"])
    for row in body_rows:
        md += "".join(f"| {_cell(is_header, text)}" for is_header, text in row) + "|\n"
    return md


def _to_markdown(elem: Node, md: deque, nested_level: int = 0) -> None:  # noqa: C901 (one branch per element type)
    def separate(sep: str = "\n\n") -> None:
        while md:
            text = md.pop().rstrip()
            if text:
                md.append(text)
                break
        if md:
            md.append(sep)

    match elem.tag:
        case "body":
            for el in elem.elements:
                _to_markdown(el, md)
        case "p":
            separate()
            md.append(elem.text or "")
            for el in elem.elements:
                _to_markdown(el, md)
            separate()
        case tag if tag in HEADING_TAGS:
            separate()
            md.append("#" * int(tag[1]) + " ")
            md.append(elem.text or "")
            for el in elem.elements:
                _to_markdown(el, md)
            separate()
        case "image":
            separate()
            separate()
        case "code":
            if elem.text:
                if elem.attrs.get("lang") or "\n" in elem.text:
                    separate()
                    md.append(f"```{elem.attrs.get('lang') or ''}\n{elem.text.strip()}\n```\n\n")
                    separate()
                else:
                    md.append(f"`{elem.text}`")
        case "list":
            separate() if nested_level == 0 else separate("\n")
            for idx, item in enumerate(elem.elements):
                prefix = _list_prefix(elem.attrs["type"], idx)
                if nested_level > 0:
                    prefix = "\t" * nested_level + prefix
                if item.text or (item.elements and item.elements[0].tag != "list"):
                    md.append(f"{prefix} ")
                _to_markdown(item, md, nested_level=nested_level)
                separate("\n")
            separate()
        case "litem":
            md.append(elem.text or "")
            for el in elem.elements:
                _to_markdown(el, md, nested_level=nested_level + 1 if el.tag == "list" else 0)
        case "data-table":
            separate()
            md.append(_table_to_markdown(elem))
            separate()
        case "a":
            link_content = elem.text or ""
            if elem.elements:
                content: deque = deque()
                for el in elem.elements:
                    _to_markdown(el, content)
                link_content += "".join(content).rstrip()
            md.append(link_content)
        case "b":
            md.append(f"**{elem.text}**" if elem.text else "")
        case "i":
            md.append(f"_{elem.text}_" if elem.text else "")
        case "text":
            md.append(elem.text or "")
        case "page-break":
            separate()
            md.append("---")
            separate()
        case _:
            raise NotImplementedError(f"Didn't handle HTML element: {elem.tag}")


def to_markdown(node: Node) -> str:
    md: deque = deque()
    _to_markdown(node, md)
    return "".join(md).rstrip()
