from collections import Counter
from typing import List

from lift_bench.hierarchical_drafting.document.document_node import HTMLTag

from lift_bench.hierarchical_drafting.document.document_factory import (
    HeaderFactory,
)
from lift_bench.hierarchical_drafting.document.document_node import (
    DocumentNode,
)
from itertools import chain


class SimpleDocumentVerbalizer:
    @classmethod
    def verbalize(cls, document_node: DocumentNode) -> str:
        """
        Verbalize the document node.
        E.g. "This document has 2 headers, 1 list and 1 file."
        """
        s = "This document has "
        c = Counter([node.type for node in document_node.get_children(recursive=True)])

        c_items = list(c.items())
        for i, (node_type, count) in enumerate(c_items):
            if i != len(c_items) - 1:
                s += f"{count} {node_type}, "
            else:
                s = s[:-2] + f" and {count} {node_type}."
        return s


class HeadersDocumentVerbalizer:
    @classmethod
    def verbalize(cls, root_node: DocumentNode) -> str:
        """
        Verbalize the document node.
        E.g. "This document has 2 headers, 1 list and 1 file."
        """
        if root_node.is_leaf():
            return "This document contains no header elements of any kind."

        s = "This document has the following structure:\n"
        header_tags: List[HTMLTag] = [HTMLTag.H1, HTMLTag.H2, HTMLTag.H3, HTMLTag.H4, HTMLTag.H5, HTMLTag.H6]
        for node in root_node.get_children(recursive=True):
            if node.type not in header_tags:
                continue
            prefix_hashtags = "#" * (header_tags.index(node.type) + 1)
            header_index = ".".join([str(idx + 1) for idx in node.get_selector() if isinstance(idx, int)])
            s += f"{prefix_hashtags} Header {header_index}\n"
        return s


class ConstraintsVerbalizer:
    @classmethod
    def verbalize(cls, root_node: DocumentNode, constraints: dict[str, dict[str, list[str]]]) -> str:
        s = ""
        for header_name, node_type_constraints in constraints.items():
            s += f"The '{header_name}' section should follow the constraints below:\n"
            for constraint_str in chain(*node_type_constraints.values()):
                s += f"\t* {constraint_str}\n"
            s += "\n"
        return s


if __name__ == "__main__":
    # verbalizer = SimpleDocumentVerbalizer()
    verbalizer = HeadersDocumentVerbalizer()

    for i in range(3):
        print(f"===== example {i} =====")
        random_tree2 = HeaderFactory.generate(max_children=3, max_depth=2)
        print(verbalizer.verbalize(random_tree2))
