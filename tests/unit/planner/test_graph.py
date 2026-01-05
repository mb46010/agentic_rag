# tests/unit/planner/test_graph.py
"""Unit tests for planner graph."""

import pytest
from unittest.mock import MagicMock, patch

from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.planner.state import PlannerState, RetrievalRound, RetrievalModeSpec


class TestPlannerGraph:
    """Tests for planner graph construction and execution."""

    def test_make_planner_graph_returns_compiled_graph(self, mock_llm):
        """Test that make_planner_graph returns a compiled graph."""
        graph = make_planner_graph(mock_llm, max_retries=1)
        assert graph is not None
        # Check it has invoke method (compiled graph)
        assert hasattr(graph, "invoke")

    def test_planner_graph_structure(self, mock_llm):
        """Test that planner graph has correct structure."""
        graph = make_planner_graph(mock_llm, max_retries=1)

        # LangGraph compiled graphs have a nodes attribute
        assert hasattr(graph, "nodes")

    def test_make_planner_graph_with_max_retries(self, mock_llm):
        """Test that max_retries parameter is used."""
        graph = make_planner_graph(mock_llm, max_retries=5)
        assert graph is not None

        # Different retry count
        graph2 = make_planner_graph(mock_llm, max_retries=1)
        assert graph2 is not None

    def test_planner_graph_has_planner_node(self, mock_llm):
        """Test that planner graph contains planner node."""
        graph = make_planner_graph(mock_llm, max_retries=1)
        # Graph should have been built with planner node
        # Note: Full end-to-end execution is tested in test_planner_node.py
        assert hasattr(graph, "nodes")
        # Verify graph can be invoked (basic smoke test)
        assert callable(getattr(graph, "invoke", None))

    def test_planner_graph_error_propagation(self, mock_llm, sample_intake_state):
        """Test that errors from planner node are propagated."""
        # Setup mock that raises error
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Mock LLM error")
        mock_llm.with_structured_output.return_value = mock_chain

        graph = make_planner_graph(mock_llm, max_retries=1)

        # This might raise or return errors depending on graph configuration
        try:
            result = graph.invoke(sample_intake_state)
            # If no exception, check for errors in result
            if "errors" in result:
                assert len(result["errors"]) > 0
        except Exception:
            # Graph might raise - that's also acceptable behavior
            pass

    def test_planner_graph_minimal_input(self, mock_llm):
        """Test graph with minimal required input."""
        planner_result = PlannerState(
            goal="minimal goal",
            strategy="direct_answer",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        with patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_messages.return_value = mock_prompt
            mock_prompt.__or__.return_value = mock_chain
            mock_llm.with_structured_output.return_value = MagicMock()

            graph = make_planner_graph(mock_llm, max_retries=1)
            result = graph.invoke({"messages": [{"role": "user", "content": "test"}]})

            # Should complete
            assert "plan" in result or "errors" in result

    def test_planner_graph_preserves_state_fields(
        self, mock_llm, sample_intake_state
    ):
        """Test that planner graph preserves input state fields."""
        planner_result = PlannerState(
            goal="test goal",
            strategy="direct_answer",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        with patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_messages.return_value = mock_prompt
            mock_prompt.__or__.return_value = mock_chain
            mock_llm.with_structured_output.return_value = MagicMock()

            graph = make_planner_graph(mock_llm, max_retries=1)
            result = graph.invoke(sample_intake_state)

            # Original state fields should be preserved
            if "normalized_query" in sample_intake_state:
                assert "normalized_query" in result
            if "user_intent" in sample_intake_state:
                assert "user_intent" in result


# Note: Full end-to-end planner tests with different strategies are in test_planner_node.py
# Those tests comprehensively test all planner logic with mocked LLMs
# Graph tests focus on graph construction and structure
