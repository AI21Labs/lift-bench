from collections import Counter

import yaml

from lift_bench.hierarchical_drafting.document.document_factory import (
    HeaderFactory,
)
from lift_bench.hierarchical_drafting.document.document_node import (
    DocumentNode,
)
from lift_bench.hierarchical_drafting.document.document_node_verbalizers import (
    SimpleDocumentVerbalizer,
)
from lift_bench.hierarchical_drafting.document.document_node import HTMLTag


def main():
    # Create a sample tree structure using allowed HTMLTag types
    root = DocumentNode(HTMLTag.H1, "Document Root")

    # Add some children using allowed types
    section1 = DocumentNode(HTMLTag.H2, "Introduction")
    section2 = DocumentNode(HTMLTag.H2, "Main Content")
    code_section = DocumentNode(HTMLTag.CODE, "sample_code.py")

    root.add_child(section1)
    root.add_child(section2)
    root.add_child(code_section)

    # Add children to sections
    para1 = DocumentNode(HTMLTag.P, "Introduction paragraph")
    para2 = DocumentNode(HTMLTag.P, "Another introduction paragraph")
    section1.add_child(para1)
    section1.add_child(para2)

    list1 = DocumentNode(HTMLTag.LIST, "Features list")
    table1 = DocumentNode(HTMLTag.DATA_TABLE, "Data summary table")
    section2.add_child(list1)
    section2.add_child(table1)

    # Demonstrate the tree
    print("Tree structure:")
    root.print_tree()

    print(f"\nRoot node: {root}")
    print(f"Tree height: {root.get_height()}")
    print(f"Number of children: {len(root.children)}")

    print("\nAll paragraphs in the tree:")
    paragraphs = root.find_nodes_with_type(HTMLTag.P, recursive=True)
    for para_node in paragraphs:
        print(f"  - {para_node.value} (depth: {para_node.get_depth()})")
        print(f"    Path: {para_node.get_selector()}")

    print("\n" + "=" * 50)
    print("YAML FUNCTIONALITY DEMONSTRATION")
    print("=" * 50)

    # Test parsing YAML string with HTMLTag compatible types
    test_yaml = """
name: Document Structure
h2:
  - name: Overview Section
    p:
      - This is a paragraph about the overview.
      - Another paragraph with more details.
    h3:
      - name: Subsection
        p:
          - Nested paragraph content.
  - name: Data Section
    data_table:
      - Summary statistics table
      - Comparison data table
    list:
      - Feature list
      - Requirements list
p:
  - Introduction paragraph
  - Conclusion paragraph
"""

    print("Parsing test YAML:")
    print(test_yaml)

    # Parse the YAML into a tree
    parsed_tree = DocumentNode.from_yaml(test_yaml, default_type=HTMLTag.H1)

    print("Parsed tree structure:")
    parsed_tree.print_tree()

    print(f"\nParsed tree height: {parsed_tree.get_height()}")
    print(f"Total paragraphs in parsed tree: {len(parsed_tree.find_nodes_with_type(HTMLTag.P, recursive=True))}")
    print(f"Total H2 sections in parsed tree: {len(parsed_tree.find_nodes_with_type(HTMLTag.H2, recursive=True))}")

    # Test edge cases
    print("\n" + "=" * 30)
    print("TESTING EDGE CASES")
    print("=" * 30)

    # Test 1: Simple string node
    simple_yaml = "simple_paragraph_content"
    simple_tree = DocumentNode.from_yaml(simple_yaml)
    print(f"\nSimple node test: {simple_tree}")

    # Test 2: Node with only name
    name_only_yaml = "name: Document Title"
    name_only_tree = DocumentNode.from_yaml(name_only_yaml, default_type=HTMLTag.H1)
    print(f"Name-only node test: {name_only_tree}")

    # Test 3: Document structure example
    doc_text = """
- name: 'Overview of Magallanes Region'
  h2:
    - 'Geographical Location'
    - 'Climate and Environment'
    - 'Economic Activities'
- name: 'History of Magallanes Region'
  h2:
    - 'Early Settlements'
    - 'Discovery and Exploration'
    - 'Modern Development'
- name: 'Demographics and Culture'
  list:
    - 'Major Ethnic Groups'
    - 'Cultural Festivals and Traditions'
- name: 'Tourism in Magallanes Region'
  h2:
    - 'Popular Tourist Attractions'
    - 'Wildlife and Nature Reserves'
    - 'Adventure Sports and Activities'
  list:
    - 'Top 5 Must-Visit Places'
- name: 'Infrastructure and Services'
  h2:
    - 'Transportation Networks'
    - 'Healthcare Facilities'
    - 'Educational Institutions'
- name: 'Challenges and Future Prospects'
  list:
    - 'Current Environmental Concerns'
    - 'Economic Growth Opportunities'
"""
    print("Document structure example:")
    print(doc_text)
    doc_yaml = yaml.safe_load(doc_text)
    doc_children = [DocumentNode.from_yaml(yaml_dict=header, default_type=HTMLTag.H1) for header in doc_yaml]
    doc_tree = DocumentNode()
    for child in doc_children:
        doc_tree.add_child(child)
    doc_tree.print_tree()

    print("\n" + "=" * 50)
    print("PRUNE FUNCTIONALITY DEMONSTRATION")
    print("=" * 50)

    # Create a sample tree for pruning demonstration using HTMLTag types
    print("\nCreating a sample tree for pruning:")
    sample_root = DocumentNode()
    documentation_header = DocumentNode(HTMLTag.H1, "Engineering Documentation")

    # Add sections
    backend_section = DocumentNode(HTMLTag.H2, "Backend Development")
    frontend_section = DocumentNode(HTMLTag.H2, "Frontend Development")
    devops_section = DocumentNode(HTMLTag.H2, "DevOps")

    documentation_header.add_child(backend_section)
    documentation_header.add_child(frontend_section)
    documentation_header.add_child(devops_section)

    # Add content to sections
    # Backend section
    backend_section.add_child(DocumentNode(HTMLTag.P, "Alice's contributions"))
    backend_section.add_child(DocumentNode(HTMLTag.P, "Bob's work"))
    backend_section.add_child(DocumentNode(HTMLTag.P, "Charlie's code"))
    backend_section.add_child(DocumentNode(HTMLTag.CODE, "Charlie's test code"))

    # Frontend section
    frontend_section.add_child(DocumentNode(HTMLTag.P, "Diana's UI work"))
    frontend_section.add_child(DocumentNode(HTMLTag.P, "Eve's components"))

    # DevOps section
    devops_section.add_child(DocumentNode(HTMLTag.P, "Frank's infrastructure"))
    devops_section.add_child(DocumentNode(HTMLTag.P, "Grace's deployment"))

    print("Original tree:")
    sample_root.print_tree()

    # Find specific content to prune to
    alice = sample_root.find_node_by_value("Alice's contributions")
    diana = sample_root.find_node_by_value("Diana's UI work")
    frank = sample_root.find_node_by_value("Frank's infrastructure")

    # Prune the tree to only include paths to specific content
    prune_target_nodes = [alice, diana, frank]
    for node in prune_target_nodes:
        if node:
            print(node.get_selector())

    target_nodes = [node for node in [alice, diana, frank] if node is not None]
    pruned_tree = sample_root.prune(target_nodes)

    print("\nPruned tree (keeping only paths to Alice, Diana, and Frank content):")
    pruned_tree.print_tree()

    print(f"\nOriginal tree has {len(sample_root.get_children(recursive=True))} total nodes")
    print(f"Pruned tree has {len(pruned_tree.get_children(recursive=True))} total nodes")

    # Demonstrate pruning to just one node
    bob = sample_root.find_node_by_value("Bob's work")
    if bob is not None:
        single_pruned = sample_root.prune([bob])
        print("\nPruned tree (keeping only path to Bob's work):")
        single_pruned.print_tree()

    # Show that original tree is unchanged
    print("\nOriginal tree remains unchanged:")
    print(f"Original tree still has {len(sample_root.get_children(recursive=True))} total nodes")

    dv = SimpleDocumentVerbalizer()
    print(dv.verbalize(sample_root))

    print("\n" + "=" * 50)
    print("HEADER FACTORY DEMONSTRATION")
    print("=" * 50)

    # Test HeaderFactory with different parameters
    print("\nGenerating random header tree with max_children=3, max_depth=4:")
    random_tree = HeaderFactory.generate(max_children=3, max_depth=4)
    random_tree.print_tree()

    print(f"\nGenerated tree height: {random_tree.get_height()}")
    print(f"Total nodes in generated tree: {len(random_tree.get_children(recursive=True)) + 1}")

    # Count header types
    node_types = [node.type for node in random_tree.get_children(recursive=True)] + [random_tree.type]
    header_counts = Counter([n for n in node_types if n is not None])
    print("Header type distribution:")
    for header_type, count in sorted(header_counts.items()):
        print(f"  {header_type}: {count}")

    print("\nGenerating another random tree with max_children=4, max_depth=2:")
    for i in range(3):
        print(f"Generating tree {i}")
        random_tree2 = HeaderFactory.generate(max_children=3, max_depth=2)
        random_tree2.print_tree()


# Example usage and demonstration
if __name__ == "__main__":
    main()
