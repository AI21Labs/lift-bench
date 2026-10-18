from enum import StrEnum
from typing import List

from lift_bench.hierarchical_drafting.metrics.registry import register_metric
from lift_bench.hierarchical_drafting.output_parsers.nested_html import ParsingError


class CaseType(StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    TITLE = "title"


WHITELIST_WORDS = [
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "as",
    "in",
    "of",
    "on",
    "to",
    "for",
    "from",
    "into",
    "like",
    "over",
    "with",
    "upon",
]


def _validate_title_case_word(w: str) -> bool:
    if w.lower() in WHITELIST_WORDS:
        return w in [w.lower(), w.title()]
    if w.lower().endswith("'s") or w.lower().endswith("'s"):
        return w[:-2] == w[:-2].title()
    elif len(w) > 1 and w[0].isdigit():
        return True
    else:
        return w == w.title()


def _inaccurate_words(s: str, case: CaseType) -> List[str]:
    if case == CaseType.LOWER:
        return [w for w in s.split() if w != w.lower()]
    elif case == CaseType.UPPER:
        return [w for w in s.split() if w != w.upper()]
    else:
        return [w for w in s.split() if not _validate_title_case_word(w)]


@register_metric("case_type")
async def case_type(completion, case_type: str) -> float:
    try:
        case = CaseType(case_type)
    except ValueError as exc:
        raise ParsingError(f"unknown case type {case_type!r}") from exc
    text = completion[-1]["content"]
    inaccurate = _inaccurate_words(text, case)
    return 1.0 if len(inaccurate) == 0 else 0.0
