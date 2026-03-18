"""Tests for graph factory function."""

from react_agent.graph import create_graph, graph


def test_create_graph_without_checkpointer():
    """Graph should compile successfully without a checkpointer."""
    g = create_graph()
    assert g is not None


def test_create_graph_with_checkpointer():
    """Graph should accept a checkpointer parameter."""
    g = create_graph(checkpointer=True)
    assert g is not None


def test_module_level_graph_exists():
    """Module-level graph variable should exist for langgraph.json compatibility."""
    assert graph is not None


def test_graph_has_summarize_node():
    """Graph should contain the summarize_conversation node."""
    g = create_graph()
    # Check that the graph has the expected nodes
    node_names = list(g.get_graph().nodes.keys())
    assert "summarize_conversation" in node_names
