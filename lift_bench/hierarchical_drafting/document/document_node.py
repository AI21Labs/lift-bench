from collections import defaultdict
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class HTMLTag(StrEnum):
    P = "p"
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H4 = "h4"
    H5 = "h5"
    H6 = "h6"
    IMAGE = "image"
    CODE = "code"
    DATA_TABLE = "data-table"
    B = "b"
    I = "i"  # noqa: E741
    TEXT_NODE = "text"
    LIST = "list"
    LIST_ITEM = "litem"
    META = "meta"
    BODY = "body"
    CUSTOM_FIELD = "custom-field"


# Define the allowed HTMLTag values
ALLOWED_HTML_TAGS = {HTMLTag.P, HTMLTag.H1, HTMLTag.H2, HTMLTag.H3, HTMLTag.CODE, HTMLTag.DATA_TABLE, HTMLTag.LIST}

# Define header tags for HeaderFactory
HEADER_TAGS = [HTMLTag.H1, HTMLTag.H2, HTMLTag.H3]
try:
    # Try to add H4, H5, H6 if they exist
    HEADER_TAGS.extend([HTMLTag.H4, HTMLTag.H5, HTMLTag.H6])
    ALLOWED_HTML_TAGS.update([HTMLTag.H4, HTMLTag.H5, HTMLTag.H6])
except AttributeError:
    # H4, H5, H6 don't exist, stick with H1-H3
    pass


class DocumentNode:
    """
    A Tree node data structure where each node has a type and a list of children nodes.
    """

    def __init__(self, node_type: Optional[HTMLTag] = None, value: str = ""):
        """
        Initialize a tree node.

        Args:
            node_type: The HTMLTag type of this node (must be one of: P, H1, H2, H3, CODE, DATA_TABLE, LIST).
                      Can be None ONLY for root nodes. If None is provided for a non-root node,
                      an exception will be raised when the node is added as a child.
            value: Optional value/data associated with this node
        """
        if node_type is not None and node_type not in ALLOWED_HTML_TAGS:
            raise ValueError(f"node_type must be one of {list(ALLOWED_HTML_TAGS)} or None, got {node_type}")

        self.type: Optional[HTMLTag] = node_type
        self.value: str = value
        self.children: List["DocumentNode"] = []
        self.parent: Optional["DocumentNode"] = None

    def add_child(self, child: "DocumentNode") -> None:
        """Add a child node to this node."""
        if child.type is None:
            raise ValueError("Cannot add a child node with node_type=None. Only root nodes can have node_type=None.")

        if child not in self.children:
            child.parent = self
            self.children.append(child)

    def remove_child(self, child: "DocumentNode") -> bool:
        """Remove a child node from this node. Returns True if successful."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False

    def get_children(self, recursive: bool = False) -> List["DocumentNode"]:
        """Get all children nodes of this node. If recursive is True, return all children recursively."""
        children = []
        for child in self.children:
            children.append(child)
            if recursive:
                children.extend(child.get_children(recursive=recursive))
        return children

    def get_children_by_type(self, recursive: bool = False) -> List[Tuple[str, List["DocumentNode"]]]:
        """Get all direct children nodes of a specific type."""
        children = self.get_children(recursive=recursive)

        dct = defaultdict(list)
        for child in children:
            if child.type is not None:  # Skip nodes with None type
                dct[child.type].append(child)
        return list([(node_type, children) for node_type, children in dct.items()])

    def find_nodes_with_type(self, node_type: HTMLTag, recursive: bool = False) -> List["DocumentNode"]:
        """Find all nodes of a specific type in the subtree rooted at this node."""
        result = []

        # Check if this node matches the type
        if self.type == node_type:
            result.append(self)

        # Check children
        for node_type_found, nodes in self.get_children_by_type(recursive=recursive):
            if node_type_found == node_type:
                result.extend(nodes)

        return result

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is a root (has no parent)."""
        return self.parent is None

    def validate_node_type(self) -> None:
        """
        Validate that the node's type configuration is valid.

        Raises:
            ValueError: If the node has type=None but is not a root node.
        """
        if self.type is None and not self.is_root():
            raise ValueError("Node with type=None must be a root node (parent must be None)")

    def get_depth(self) -> int:
        """Get the depth of this node from the root (root has depth 0)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def get_height(self) -> int:
        """Get the height of the subtree rooted at this node."""
        if self.is_leaf():
            return 0
        return 1 + max(child.get_height() for child in self.children)

    def get_index_in_parent(self) -> int:
        """Get the index of this node in its parent's children list, out of the children with the same type."""
        if self.parent is None:
            return -1
        same_type_children = [child for child in self.parent.children if child.type == self.type]
        return same_type_children.index(self)

    def get_selector(self) -> List[str]:
        """Get the path from this node to a target node."""
        path = []
        current = self
        while current.parent is not None:
            path = [current.type, current.get_index_in_parent()] + path
            current = current.parent
        return path

    @property
    def alias(self) -> str:
        """
        Return either self.value if value != '', otherwise return '<node_type> <hierarchy_position>'.
        hierarchy_position is derived from the current node's position under header ancestors,
        followed by the index of the immediate parent.

        Returns:
            str: The alias for this node
        """
        # If the node has a non-empty value, return it
        if self.value != "":
            return self.value

        # Otherwise, generate hierarchical position
        node_type_name = self._get_node_type_name()
        hierarchy_position = self._get_hierarchy_position()

        if self.is_root():
            return node_type_name
        else:
            return f"{node_type_name} {hierarchy_position}"

    @property
    def ancestors_alias_list(self) -> list[str]:
        """
        Return list of aliases of the full ancestry path.
        """
        # If the node has a non-empty value, return it
        ancestors_aliases = [a.alias for a in self.get_ancestors() if not a.is_root()]
        return ancestors_aliases + [self.alias]

    def _get_node_type_name(self) -> str:
        """Get a human-readable name for the node type."""
        if self.is_root():
            return "Root"

        type_mapping = {
            HTMLTag.H1: "Header",
            HTMLTag.H2: "Header",
            HTMLTag.H3: "Header",
            HTMLTag.H4: "Header",
            HTMLTag.H5: "Header",
            HTMLTag.H6: "Header",
            HTMLTag.P: "Paragraph",
            HTMLTag.CODE: "Code Block",
            HTMLTag.DATA_TABLE: "Table",
            HTMLTag.LIST: "List",
        }

        return type_mapping.get(self.type, str(self.type))

    def _get_hierarchy_position(self) -> str:
        """
        Get the hierarchical position string (e.g., '1.2.1').
        Based on header ancestors' positions followed by current node's index.

        Returns:
            str: Hierarchy position like '1.2.1'
        """
        position_parts = []

        # Collect header ancestors and their positions
        ancestors = self.get_ancestors()

        # Get positions for header ancestors
        for ancestor in ancestors:
            if ancestor.type in HEADER_TAGS:
                # Get index among siblings of the same header type (1-based)
                header_index = ancestor.get_index_in_parent() + 1
                position_parts.append(str(header_index))

        # Add current node's index among ALL siblings (1-based), not just same type
        if self.parent is not None:
            # Get index among all siblings regardless of type
            current_index = self.parent.children.index(self) + 1
            position_parts.append(str(current_index))
        else:
            # Root node
            position_parts.append("1")

        return ".".join(position_parts) if position_parts else "1"

    def get_ancestors(self):
        """
        Get all ancestors of this node. Return them as a list, starting from the root node up to the current node.

        Returns:
            list: List of ancestors of this node
        """
        current = self
        ancestors = []
        while current.parent is not None:
            ancestors.append(current.parent)
            current = current.parent
        # Reverse to get root-to-node order
        ancestors.reverse()
        return ancestors

    @staticmethod
    def from_yaml(
        yaml_string: Optional[str] = None,
        yaml_dict: Optional[Dict[str, Any]] = None,
        default_type: HTMLTag = HTMLTag.H1,
    ) -> "DocumentNode":
        """
        Parse a YAML string into a TreeNode object.

        Each YAML element can be either:
        1. A string - representing the node's name with no children
        2. A YAML object with:
           - name: <node_name> - the name of the node
           - <node_type_1>: [list of child nodes]
           - <node_type_2>: [list of child nodes]

        Args:
            yaml_string: The YAML string to parse
            yaml_dict: The YAML dictionary to parse
            default_type: Default HTMLTag type for nodes (must be one of the allowed types)

        Returns:
            TreeNode: The root node of the parsed tree
        """
        if default_type not in ALLOWED_HTML_TAGS:
            raise ValueError(f"default_type must be one of {list(ALLOWED_HTML_TAGS)}, got {default_type}")

        assert (yaml_string is not None) != (yaml_dict is not None), "Either yaml_string or yaml_dict must be provided"

        if yaml_dict is None and yaml_string is not None:
            try:
                yaml_dict = yaml.safe_load(yaml_string)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML: {e}")

        if yaml_dict is None:
            raise ValueError("Unable to parse YAML data")

        return DocumentNode._parse_yaml_node(yaml_dict, default_type=default_type)

    @staticmethod
    def _parse_yaml_node(data: Union[str, Dict[str, Any]], default_type: HTMLTag = HTMLTag.H1) -> "DocumentNode":
        """
        Recursively parse a YAML node into a TreeNode.

        Args:
            data: Either a string (leaf node) or dict (node with children)
            default_type: Default HTMLTag type for nodes

        Returns:
            TreeNode: The parsed node
        """
        if default_type not in ALLOWED_HTML_TAGS:
            raise ValueError(f"default_type must be one of {list(ALLOWED_HTML_TAGS)}, got {default_type}")

        # Case 1: Simple string node (leaf)
        if isinstance(data, str):
            # Use 'P' as default type for string nodes (paragraphs)
            return DocumentNode(node_type=HTMLTag.P, value=data)

        # Case 2: Dictionary with name and children
        if isinstance(data, dict):
            # Extract the name
            if "name" not in data:
                raise ValueError("YAML node object must have a 'name' field")

            name = data["name"]
            # Use default_type for named nodes
            node = DocumentNode(node_type=default_type, value=name)

            # Process children by type
            for node_type_str, children_list in data.items():
                if node_type_str == "name":
                    continue  # Skip the name field

                # Convert string to HTMLTag
                try:
                    child_type = getattr(HTMLTag, node_type_str.upper())
                    if child_type not in ALLOWED_HTML_TAGS:
                        raise ValueError(f"Child node type {child_type} is not allowed")
                except AttributeError:
                    raise ValueError(f"Invalid HTMLTag: {node_type_str}")

                if not isinstance(children_list, list):
                    raise ValueError(f"Children for type '{node_type_str}' must be a list")

                # Parse each child
                for child_data in children_list:
                    child_node = DocumentNode._parse_yaml_node(child_data, default_type=child_type)
                    # Override the child's type with the specified type
                    child_node.type = child_type
                    node.add_child(child_node)

            return node

        else:
            raise ValueError(f"Invalid YAML node data type: {type(data)}")

    def prune(self, target_nodes: List["DocumentNode"]) -> "DocumentNode":
        """
        Prune the tree to include only paths that lead to the target nodes.

        Args:
            target_nodes: List of nodes that should be preserved as leaves (or ancestors of leaves)

        Returns:
            TreeNode: A new pruned tree where only leaves are a subset of target_nodes
        """
        # Create a set for faster lookup
        target_set = set(target_nodes)

        # Create the pruned tree starting from this node
        pruned_root = self._prune_recursive(target_set)

        # If the root itself gets pruned (shouldn't happen in normal cases),
        # return an empty tree with the same type and value
        if pruned_root is None:
            return DocumentNode(node_type=self.type, value=self.value)

        return pruned_root

    def _prune_recursive(self, target_set: set) -> Optional["DocumentNode"]:
        """
        Recursively prune the tree to keep only paths leading to target nodes.

        Args:
            target_set: Set of target nodes to preserve

        Returns:
            TreeNode or None: The pruned node if it should be kept, None if it should be removed
        """
        # Check if this node or any of its descendants should be kept
        should_keep_node = self._should_keep_node(target_set)

        if not should_keep_node:
            return None

        # Create a copy of this node
        pruned_node = DocumentNode(node_type=self.type, value=self.value)

        # Recursively prune children
        for child in self.children:
            pruned_child = child._prune_recursive(target_set)
            if pruned_child is not None:
                pruned_node.add_child(pruned_child)

        return pruned_node

    def _should_keep_node(self, target_set: set) -> bool:
        """
        Determine if this node should be kept in the pruned tree.
        A node should be kept if:
        1. It's in the target set, OR
        2. Any of its descendants is in the target set

        Args:
            target_set: Set of target nodes

        Returns:
            bool: True if the node should be kept
        """
        # Check if this node is a target node
        if self in target_set:
            return True

        # Check if any descendant is a target node
        for child in self.children:
            if child._should_keep_node(target_set):
                return True

        return False

    def find_node_by_value(self, value: Any) -> Optional["DocumentNode"]:
        """
        Find a node in the tree by its value.

        Args:
            value: The value to search for

        Returns:
            TreeNode or None: The first node found with the given value, or None if not found
        """
        if self.value == value:
            return self

        for child in self.children:
            result = child.find_node_by_value(value)
            if result is not None:
                return result

        return None

    def compare(self, other: "DocumentNode") -> bool:
        """
        Compare this DocumentNode with another DocumentNode.

        Args:
            other: The other DocumentNode to compare with
            strict: If True, compare children in their existing order.
                   If False, just check that the counts of children types are equal.

        Returns:
            bool: True if the nodes have the same structure, False otherwise
        """
        # First check if both nodes have the same type
        if self.type != other.type:
            return False

        # Strict comparison: same order, same types
        if len(self.children) != len(other.children):
            return False

        # Compare each child in order
        for self_child, other_child in zip(self.children, other.children):
            if not self_child.compare(other_child):
                return False

        return True

    @staticmethod
    def from_md(markdown_text: str, only_skeleton=False) -> "DocumentNode":
        """
        Parse markdown headers into a DocumentNode tree structure.

        Args:
            markdown_text: Markdown text with headers (# ## ### etc.)
            only_skeleton: If True, ignore node value.

        Returns:
            DocumentNode: Root node of the parsed tree
        """
        lines = markdown_text.strip().split("\n")
        root = DocumentNode()  # Root node with no type
        stack = [root]  # Stack to keep track of current path

        for line in lines:
            line = line.strip()
            if not line or not line.startswith("#"):
                continue

            # Count header level
            level = 0
            while level < len(line) and line[level] == "#":
                level += 1

            if level == 0:
                continue

            # Get header text
            header_text = line[level:].strip()

            # Determine HTMLTag based on level
            if level == 1:
                tag = HTMLTag.H1
            elif level == 2:
                tag = HTMLTag.H2
            elif level == 3:
                tag = HTMLTag.H3
            elif level == 4 and hasattr(HTMLTag, "H4"):
                tag = HTMLTag.H4
            elif level == 5 and hasattr(HTMLTag, "H5"):
                tag = HTMLTag.H5
            elif level == 6 and hasattr(HTMLTag, "H6"):
                tag = HTMLTag.H6
            else:
                # Default to H3 for levels > 3 if H4-H6 don't exist
                tag = HTMLTag.H3

            # Create new node
            if only_skeleton:
                node = DocumentNode(node_type=tag)
            else:
                node = DocumentNode(node_type=tag, value=header_text)

            # Find the correct parent (pop stack until we find appropriate level)
            while len(stack) > level:
                stack.pop()

            # Add as child to the current parent
            stack[-1].add_child(node)
            stack.append(node)

        return root

    def to_md(self) -> str:
        """
        Convert the DocumentNode tree to markdown format.

        Returns:
            str: Markdown representation of the tree
        """
        lines = []
        self._to_md_recursive(lines)
        return "\n".join(lines)

    def _to_md_recursive(self, lines: List[str]) -> None:
        """
        Recursively convert nodes to markdown lines.

        Args:
            lines: List to append markdown lines to
        """
        # Only process header nodes (not root or other types)
        if self.type in HEADER_TAGS:
            # Determine number of # based on header type
            if self.type == HTMLTag.H1:
                header_level = 1
            elif self.type == HTMLTag.H2:
                header_level = 2
            elif self.type == HTMLTag.H3:
                header_level = 3
            elif hasattr(HTMLTag, "H4") and self.type == HTMLTag.H4:
                header_level = 4
            elif hasattr(HTMLTag, "H5") and self.type == HTMLTag.H5:
                header_level = 5
            elif hasattr(HTMLTag, "H6") and self.type == HTMLTag.H6:
                header_level = 6
            else:
                header_level = 3  # Default

            # Create markdown header
            header_prefix = "#" * header_level
            lines.append(f"{header_prefix} {self.alias}")

        # Process children
        for child in self.children:
            child._to_md_recursive(lines)

    def __str__(self) -> str:
        """String representation of the node."""
        # value_str = f" (value: {self.value})" if self.value is not None else ""
        # return f"TreeNode(type: {self.type}{value_str}, children: {len(self.children)})"
        if self.type is None:
            node_str = "ROOT"
        else:
            node_str = self.alias
        # elif self.value == "":
        #     node_str = f"{str(self.type)}-header"
        # else:
        #     node_str = f"{str(self.type)}-header: {self.value}"
        return f"{node_str} (children: {len(self.children)})"

    def __repr__(self) -> str:
        return self.__str__()

    def print_tree(self, indent: str = "", is_last: bool = True) -> None:
        """Pretty print the tree structure starting from this node."""
        connector = "└── " if is_last else "├── "
        print(f"{indent}{connector}{self}")

        # Update indent for children
        extension = "    " if is_last else "│   "
        new_indent = indent + extension

        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            child.print_tree(new_indent, is_last_child)
