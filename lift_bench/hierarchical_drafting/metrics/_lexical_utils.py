import re
from enum import StrEnum
from typing import Iterator

from lift_bench.hierarchical_drafting.output_parsers.html import (
    HEADING_TAGS,
    Node,
    parse_markdown,
)

# Text elements left out of the "without headings" variants.
_EXCLUDED_FROM_LEXICAL_MATCH = ("image", *HEADING_TAGS)


class MatchType(StrEnum):
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    EXACT = "exact"
    REGEX = "regex"


def _find_lexical_match(
    text: str,
    string: str,
    match_type: str,
    fuzzy_match: bool = True,
    strict_word_boundaries: bool = True,
) -> int:
    if fuzzy_match:
        string = string.strip().rstrip(".")
        text = text.strip().rstrip(".")
    case_flag = re.RegexFlag.I if fuzzy_match else 0
    if match_type == MatchType.EXACT:
        m = re.match(re.escape(string), text, flags=case_flag)
        return 1 if (m and m.start() == 0 and m.end() == len(text)) else 0
    if match_type == MatchType.REGEX:
        return len(re.findall(string, text, flags=case_flag))
    prefix = r"^" if match_type == MatchType.STARTS_WITH else (r"(?<![a-zA-Z0-9_])" if strict_word_boundaries else "")
    suffix = r"$" if match_type == MatchType.ENDS_WITH else (r"(?![a-zA-Z0-9_])" if strict_word_boundaries else "")
    pattern = prefix + re.escape(string) + suffix
    return len(re.findall(pattern, text, flags=case_flag))


def _xml_children(node: Node) -> list[tuple[str, str | None, list]]:
    """A node's children as (tag, text, children) in the order its XML serialization lists them."""
    if node.tag == "data-table":
        rows = [("thead", node.attrs["head"]), ("tbody", node.attrs["body"])]
        return [
            (section, None, [("trow", None, [("th" if h else "td", t, []) for h, t in row]) for row in section_rows])
            for section, section_rows in rows
            if section_rows
        ]
    return [(el.tag, el.text, _xml_children(el)) for el in node.elements]


def _textual_elements(markdown: str) -> list[tuple[str, str]]:
    """(tag, text) of every element of the document's HTML tree that holds text, in document order.

    A `text` element first under its parent carries the parent's own text (e.g. a heading's title), so it is tagged
    with the parent's tag.
    """
    body = parse_markdown(markdown)

    def walk(parent_tag: str, children: list) -> Iterator[tuple[str, str]]:
        for index, (tag, text, grandchildren) in enumerate(children):
            if text:
                yield (parent_tag if tag == "text" and index == 0 else tag), text
            yield from walk(tag, grandchildren)

    return list(walk("body", _xml_children(body)))


def _texts_to_match(text: str) -> list[str]:
    """The text itself, then its parsed text elements joined with "" and "\n", without and with headings/images."""
    elements = _textual_elements(text)
    all_elements = [t for _, t in elements]
    body_elements = [t for tag, t in elements if tag not in _EXCLUDED_FROM_LEXICAL_MATCH]
    return [text, "".join(body_elements), "".join(all_elements), "\n".join(body_elements), "\n".join(all_elements)]


def lexical_match_count(
    text: str,
    string: str,
    match_type: str,
    fuzzy_match: bool = True,
    strict_word_boundaries: bool = True,
) -> int:
    """Number of matches of `string` in `text`.

    A fuzzy match is case-insensitive and also tries `string` with a lowercase first letter, against the text and
    against its parsed markdown text (formatting markers, line wrapping and escapes undone): the first variant
    that matches gives the count.
    """
    if fuzzy_match:
        for candidate_text in _texts_to_match(text):
            for s in [string, string[0].lower() + string[1:]]:
                count = _find_lexical_match(candidate_text, s, match_type, fuzzy_match, strict_word_boundaries)
                if count > 0:
                    return count
        return 0
    return _find_lexical_match(text, string, match_type, fuzzy_match, strict_word_boundaries)


def check_bound_validity(value: int, lower_bound: int | None, upper_bound: int | None) -> bool:
    if lower_bound is None:
        return value <= upper_bound
    if upper_bound is None:
        return value >= lower_bound
    return lower_bound <= value <= upper_bound
