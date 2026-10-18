import hashlib
import random

import inflect

from lift_bench.hierarchical_drafting.constraints.base import (
    VerifiableConstraintMetadata,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.base import (
    TaskVerbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.registry import (
    register_task_verbalizer,
)
from lift_bench.hierarchical_drafting.pipeline.task_verbalization.utils import (
    render_constraint_text,
)

_inflect = inflect.engine()

# Connectors that keep constraints inside the current sentence.
_CLAUSE_CONNECTORS: list[str] = [
    ", ",
    " and ",
    ", and ",
    ", also ",
    ", in addition ",
]

# Connectors that close the current sentence and open a new one.
_SENTENCE_STARTERS: list[str] = [
    "Also, ",
    "Furthermore, ",
    "Additionally, ",
    "Moreover, ",
    "In addition, ",
    "At the same time, ",
    "As an additional requirement, ",
    "As another condition, ",
]

_ALL_INLINE_CONNECTORS: list[str] = _CLAUSE_CONNECTORS + _SENTENCE_STARTERS

# Introductory phrases for each grouped artifact section ("include" is appended by the renderer).
_ARTIFACT_SECTION_HEADERS: list[str] = [
    "In addition, include",
    "Also, include",
    "Additionally, include",
    "Furthermore, include",
]


def _lowercase_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _to_singular(word: str) -> str:
    singular = _inflect.singular_noun(word)
    return singular if singular else word


def _to_plural(word: str) -> str:
    # plural_noun() has bidirectional behaviour: normalise to singular first so it always
    # gets a predictable input, regardless of whether the type name arrives singular or plural.
    return _inflect.plural_noun(_to_singular(word))


@register_task_verbalizer("ifeval_flat")
class IFEvalFlatTaskVerbalizer(TaskVerbalizer):
    """IFEval-style verbalizer for flat tasks.

    All constraint units — individual inline constraints and per-type artifact groups —
    are shuffled together and rendered in that mixed order.  Inline constraints are joined
    with randomly chosen clause or sentence-break connectors.  Artifact groups are always
    their own numbered bullet block.

    The opening instruction is taken from ``prompt_variations`` (generated upstream by
    ``PromptVariationsBlock``) when available, falling back to ``"Write about {topic}"``.
    """

    def __init__(self):
        self.artifact_constraint_types = [
            "code_snippets",
            "list",
            "tables",
        ]  # TODO(eg): align with constraint_name values from pipeline blocks

    async def build_messages(
        self,
        topic: str,
        document_outline: str,
        header_to_constraints_dct: dict,
        use_strict_header_names: bool,
        verbalized_topic: str | None = None,
    ) -> list[dict]:
        units = self._split_and_verbalize(header_to_constraints_dct)
        rng = random.Random(hashlib.md5(topic.encode()).hexdigest())
        rng.shuffle(units)

        opening, opens_new_sentence = self._prepare_opening(verbalized_topic, topic, rng)
        text = self._assemble_units(units, opening, opens_new_sentence, rng)
        return [{"role": "user", "content": text}]

    def _prepare_opening(self, verbalized_topic: str | None, topic: str, rng: random.Random) -> tuple[str, bool]:
        """Return (opening_text, opens_new_sentence).

        opens_new_sentence is True when the first inline constraint should begin a new
        sentence rather than be joined inline to the opening.
        """
        opening = verbalized_topic or f"Write about {topic}"
        opens_new_sentence = any(opening.endswith(c) for c in ".?!")
        # Randomly strip trailing punctuation so the first constraint can be joined inline.
        if opens_new_sentence and rng.random() < 0.5 and opening[-1] in ".!":
            opening = opening[:-1]
            opens_new_sentence = False
        return opening, opens_new_sentence

    def _render_artifact_section(self, ctype: str, constraints: list[str], rng: random.Random) -> str:
        """Render an artifact constraint group as a numbered bullet block."""
        if len(constraints) == 1:
            return constraints[0]
        type_name = ctype.replace("_", " ").replace("-", " ")
        label = f"{len(constraints)} {_to_plural(type_name)}"
        header = rng.choice(_ARTIFACT_SECTION_HEADERS)
        bullets = "\n".join(f"{i}. {c}" for i, c in enumerate(constraints, start=1))
        return f"{header} {label}:\n{bullets}"

    def _assemble_units(
        self,
        units: list[tuple],
        opening: str,
        opens_new_sentence: bool,
        rng: random.Random,
    ) -> str:
        """Join constraint units into a final prompt string."""
        parts: list[str] = []
        sentence: list[str] = [opening]

        def flush() -> None:
            if sentence:
                text = "".join(sentence)
                if text[-1] not in ".?!":
                    text += "."
                parts.append(text)
                sentence.clear()

        for i, (unit_type, ctype, content) in enumerate(units):
            if unit_type == "artifact":
                flush()
                parts.append(self._render_artifact_section(ctype, list(content), rng))
                continue

            # Inline constraint: pick a connector then decide whether to break first.
            text = _lowercase_first(content)
            if not sentence:
                # Resuming after an artifact block or sentence break.
                sentence.append(rng.choice(_SENTENCE_STARTERS) + text)
            elif i == 0 and opens_new_sentence:
                # Opening ended with punctuation; treat this as a new sentence appended.
                sentence.append(" " + rng.choice(_SENTENCE_STARTERS) + text)
            elif i == 0:
                # Opening had no terminal punctuation; join inline.
                sentence.append(rng.choice(_CLAUSE_CONNECTORS) + text)
            else:
                connector = rng.choice(_ALL_INLINE_CONNECTORS)
                if connector in _SENTENCE_STARTERS:
                    flush()
                sentence.append(connector + text)

        flush()
        if parts and parts[-1][-1] not in ".?!":
            parts[-1] += "."
        return "\n".join(parts)

    def _split_and_verbalize(self, header_to_constraints_dct: dict) -> list[tuple]:
        """Return a flat list of (unit_type, ctype, content) units ready for shuffling.

        unit_type is "inline" or "artifact"; content is a str for inline units
        and list[str] for artifact groups.
        """
        inline_units: list[tuple] = []
        artifact_texts: dict[str, list[str]] = {}

        for header_constraints in header_to_constraints_dct.values():
            for constraint_type, constraint_json_lst in header_constraints.items():
                if constraint_type == "doc_structure":
                    continue
                for constraint_json in constraint_json_lst:
                    metadata = VerifiableConstraintMetadata.from_json(constraint_json)
                    text = render_constraint_text(metadata).rstrip(".")
                    if constraint_type in self.artifact_constraint_types:
                        artifact_texts.setdefault(constraint_type, []).append(text)
                    else:
                        inline_units.append(("inline", constraint_type, text))

        artifact_units = [("artifact", ctype, texts) for ctype, texts in artifact_texts.items()]
        return inline_units + artifact_units
