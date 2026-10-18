import random

import argparse
from tqdm.auto import trange

from lift_bench.hierarchical_drafting.document.document_node import (
    DocumentNode,
    HEADER_TAGS,
)


class HeaderFactory:
    """Factory class for generating random header-based document trees."""

    @staticmethod
    def generate(max_children: int, max_depth: int) -> DocumentNode:
        """
        Generate a random DocumentNode tree consisting only of header nodes (H1-H6).

        Args:
            max_children: Maximum number of children per node (1 to max_children).
                         If a node would have 1 child, it gets 0 children instead.
            max_depth: Maximum depth of the tree (won't be greater than 6).

        Returns:
            DocumentNode: A randomly generated tree with header nodes only.
        """
        if max_depth == 0 or max_children == 0:
            return DocumentNode(node_type=HEADER_TAGS[0])

        # Start with H1 as root
        root = DocumentNode()
        HeaderFactory._generate_children(root, 0, max_children, max_depth)
        return root

    @staticmethod
    def _create_header_node(depth: int) -> DocumentNode:
        """Create a header node based on depth level."""
        # Map depth to header level (0->H1, 1->H2, etc.)
        header_level = min(depth, len(HEADER_TAGS) - 1)
        header_tag = HEADER_TAGS[header_level]

        return DocumentNode(node_type=header_tag)

    @staticmethod
    def _generate_children(node: DocumentNode, current_depth: int, max_children: int, max_depth: int) -> None:
        """Recursively generate children for a node."""
        # Stop if we've reached max depth
        if current_depth >= max_depth:
            return

        # Determine number of children (1 to max_children, but avoid exactly 1)
        num_children = random.randint(1, max_children)
        if num_children == 1 and current_depth != 0:
            num_children = 0  # If it would be 1 child, make it 0 instead

        # Create children
        for i in range(num_children):
            child = HeaderFactory._create_header_node(current_depth)
            node.add_child(child)

            # Recursively generate children for this child
            HeaderFactory._generate_children(child, current_depth + 1, max_children, max_depth)


def main():
    for _ in trange(args.num_docs):
        doc = HeaderFactory.generate(args.max_children, args.max_depth)
        print(doc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=4)
    ap.add_argument("--max-children", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    main()
