"""Marko extension for generalized list markers.

Besides `-`, `*`, `+` and decimals, lines starting with a letter or a roman numeral followed by `.` or `)` open a
list, and the list's marker type is rendered into the HTML so the list keeps it.
"""

import re
from enum import StrEnum
from typing import NamedTuple, cast

from marko import Markdown, MarkoExtension, block
from marko.ext.gfm import GFM
from marko.source import Source


class ListType(StrEnum):
    ORDERED_DECIMALS = "decimals"
    ORDERED_UPPERCASE_ROMAN_NUMERALS = "roman_upper"
    ORDERED_LOWERCASE_ROMAN_NUMERALS = "roman_lower"
    ORDERED_UPPERCASE_LETTERS = "upper"
    ORDERED_LOWERCASE_LETTERS = "lower"
    UNORDERED_BULLETS = "bullets"
    UNORDERED_DASHES = "dashes"
    UNORDERED_PLUSES = "pluses"

    @property
    def is_ordered(self) -> bool:
        return self not in (ListType.UNORDERED_BULLETS, ListType.UNORDERED_DASHES, ListType.UNORDERED_PLUSES)


LIST_TYPE_TO_MARKER_STR = {
    ListType.UNORDERED_BULLETS: "*",
    ListType.UNORDERED_DASHES: "-",
    ListType.UNORDERED_PLUSES: "+",
    ListType.ORDERED_DECIMALS: "1",
    ListType.ORDERED_LOWERCASE_LETTERS: "a",
    ListType.ORDERED_LOWERCASE_ROMAN_NUMERALS: "i",
    ListType.ORDERED_UPPERCASE_LETTERS: "A",
    ListType.ORDERED_UPPERCASE_ROMAN_NUMERALS: "I",
}
MARKER_STR_TO_LIST_TYPE = {v: k for k, v in LIST_TYPE_TO_MARKER_STR.items()}


_GENERALIZED_LIST_MARKER_PATTERN = r" {,3}(\w[.)]|[IXVL]{1,8}[.)]|[ixvl]{1,8}[.)]|\d{1,9}[.)]|[*\-+])[ \t\n\r\f]"


def _resolve_list_type(bullet: str) -> ListType:
    bullet = bullet.replace(".", "").replace(")", "").strip()
    if bullet in MARKER_STR_TO_LIST_TYPE:
        return MARKER_STR_TO_LIST_TYPE[bullet]
    if bullet.isdigit():
        return ListType.ORDERED_DECIMALS
    if len(bullet) == 1 and bullet.isalpha() and bullet.isupper() and bullet not in "IXVL":
        return ListType.ORDERED_UPPERCASE_LETTERS
    if len(bullet) == 1 and bullet.isalpha() and bullet.islower() and bullet not in "ixvl":
        return ListType.ORDERED_LOWERCASE_LETTERS
    if all(b in "IXVL" for b in bullet):
        return ListType.ORDERED_UPPERCASE_ROMAN_NUMERALS
    if all(b in "ixvl" for b in bullet):
        return ListType.ORDERED_LOWERCASE_ROMAN_NUMERALS
    raise ValueError("Unsupported bullet", bullet)


def _bullet_consistent_with_list_type(bullet: str, list_type: ListType) -> bool:
    bullet = bullet.replace(".", "").replace(")", "").strip()
    match bullet:
        case "*" | "-":
            return list_type == ListType.UNORDERED_BULLETS
        case x if x.isdigit():
            return list_type == ListType.ORDERED_DECIMALS
        case x if x.isupper():
            return list_type in (ListType.ORDERED_UPPERCASE_LETTERS, ListType.ORDERED_UPPERCASE_ROMAN_NUMERALS)
        case x if x.islower():
            return list_type in (ListType.ORDERED_LOWERCASE_LETTERS, ListType.ORDERED_LOWERCASE_ROMAN_NUMERALS)
        case _:
            raise ValueError("Unsupported bullet", bullet)


class _GeneralizedListInfo(NamedTuple):
    bullet: str
    ordered: bool
    start: int
    list_type: ListType


class _GeneralizedList(block.List):
    pattern = _GENERALIZED_LIST_MARKER_PATTERN
    override = True

    def __init__(self, info: _GeneralizedListInfo) -> None:  # noqa: D107
        self.bullet, self.ordered, self.start, self.list_type = info
        self.tight = True

    @classmethod
    def match(cls, source: Source) -> bool:
        m = source.expect_re(cls.pattern)
        if not m:
            return False
        bullet = m.group(1)
        try:
            list_type = _resolve_list_type(bullet)
        except ValueError:
            return False
        start = int(bullet[:-1]) if list_type == ListType.ORDERED_DECIMALS else 1
        source.context.list_info = _GeneralizedListInfo(bullet, list_type.is_ordered, start, list_type)
        return True


class _GeneralizedListItem(block.ListItem):
    pattern = _GENERALIZED_LIST_MARKER_PATTERN
    override = True

    @classmethod
    def match(cls, source: Source) -> bool:
        if source.parser.block_elements["ThematicBreak"].match(source):
            return False
        if not source.expect_re(cls.pattern):
            return False
        next_line = cast(str, source.next_line(False)).expandtabs(4)
        prefix_pos = 0
        m = re.match(source.prefix, next_line)
        if m is not None:
            prefix_pos = m.end()
        indent, bullet, mid, _ = cls.parse_leading(next_line.rstrip(), prefix_pos)
        parent = source.state
        assert isinstance(parent, _GeneralizedList)
        if (parent.ordered and not _bullet_consistent_with_list_type(bullet, parent.list_type)) or bullet[
            -1
        ] != parent.bullet[-1]:
            return False
        if not parent.ordered and bullet != parent.bullet:
            return False
        source.context.list_item_info = cls.ParseInfo(indent, bullet, mid)
        return True


class _GeneralizedListRenderMixin:
    def render_list(self, element: block.List) -> str:
        list_type = cast(_GeneralizedList, element).list_type
        tag = "ol" if element.ordered else "ul"
        extra = f' start="{element.start}"' if element.ordered and element.start != 1 else ""
        extra += f' marker_type="{LIST_TYPE_TO_MARKER_STR[list_type]}"'
        return f"<{tag}{extra}>\n{self.render_children(element)}</{tag}>\n"  # type: ignore[attr-defined]


markdown = Markdown(
    extensions=[
        GFM,
        MarkoExtension(
            elements=[_GeneralizedList, _GeneralizedListItem],
            renderer_mixins=[_GeneralizedListRenderMixin],
        ),
    ]
)
