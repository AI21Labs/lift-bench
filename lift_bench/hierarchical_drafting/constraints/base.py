from abc import ABC
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from lift_bench.hierarchical_drafting.document.document_node import HTMLTag
from lift_bench.hierarchical_drafting.utils.polymorphic_dataclass import (
    polymorphic_baseclass,
    polymorphic_childclass,
)


@polymorphic_baseclass(type_field_name="constraint_name")
@dataclass(
    kw_only=True
)  # see here for kw_only: https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class Constraint(ABC):
    # This is to remove annoying warnings about this method being missing
    def to_json(self) -> str:
        # See implementation in the polymorphic_dataclass module
        raise NotImplementedError

    @staticmethod
    def from_json(json_str: str) -> "Constraint":
        # See implementation in the polymorphic_dataclass module
        raise NotImplementedError


@polymorphic_childclass(type_field_value="verifiable_constraint")
@dataclass(kw_only=True)
class VerifiableConstraint(Constraint):
    def to_evaluation_dict(self):
        raise NotImplementedError()


@dataclass_json
@dataclass(kw_only=True)
class VerifiableConstraintMetadata:
    constraint: VerifiableConstraint
    artifact_html_tag: Optional[
        HTMLTag
    ]  # sub HTML element on which the constraint is applied (if None, apply constraint on base HTML element)
    artifact_html_tag_idx: Optional[int] = (
        None  # check constraint on a specific HTML tag instance (if None, an incremental index will be used per HTML tag path)
    )
    check_title: bool  # indicator for applying constraint on content/title of HTML element
    custom_text: Optional[str] = None  # if set, used as the verbalized constraint text instead of the rigid template

    def __post_init__(self):
        if self.artifact_html_tag is None:
            assert self.artifact_html_tag_idx is None, (
                "If artifact_html_tag is None, artifact_html_tag_idx must be None"
            )

    # This is to remove annoying warnings about this method being missing
    def to_json(self) -> str:
        # See implementation in the polymorphic_dataclass module
        raise NotImplementedError

    @staticmethod
    def from_json(json_str: str) -> "VerifiableConstraintMetadata":
        # See implementation in the polymorphic_dataclass module
        raise NotImplementedError
