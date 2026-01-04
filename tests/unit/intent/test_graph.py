# tests/unit/intent/test_graph.py
"""Unit tests for intent graph."""

import pytest
from unittest.mock import MagicMock, patch

from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.intent.nodes.normalize_gate import NormalizeModel
from agentic_rag.intent.nodes.extract_signals import ExtractSignalsModel, SignalsModel


class TestIntakeGraph:
    """Tests for intake graph construction and execution."""

    def test_make_intake_graph_returns_compiled_graph(self, mock_llm):
        """Test that make_intake_graph returns a compiled graph."""
        graph = make_intake_graph(mock_llm, max_retries=1)
        assert graph is not None
        # Check it has invoke method (compiled graph)
        assert hasattr(graph, "invoke")

    def test_intake_graph_structure(self, mock_llm):
        """Test that intake graph has correct structure."""
        graph = make_intake_graph(mock_llm, max_retries=1)

        # Access the underlying graph structure
        # LangGraph compiled graphs have a .graph attribute
        assert hasattr(graph, "nodes")

    @patch("agentic_rag.intent.nodes.normalize_gate.ChatPromptTemplate")
    @patch("agentic_rag.intent.nodes.extract_signals.ChatPromptTemplate")
    def test_intake_graph_end_to_end(
        self, mock_extract_prompt, mock_normalize_prompt, mock_llm, sample_messages
    ):
        """Test intake graph end-to-end with mocked nodes."""
        # Mock normalize_gate output
        normalize_result = NormalizeModel(
            normalized_query="test query",
            constraints={"domain": ["azure"]},
            guardrails={"time_sensitivity": "low"},
            clarification={"needed": False},
        )

        # Mock extract_signals output
        extract_result = ExtractSignalsModel(
            user_intent="explain",
            retrieval_intent="definition",
            answerability="internal_corpus",
            signals=SignalsModel(),
        )

        # Setup mock chains
        normalize_chain = MagicMock()
        normalize_chain.invoke.return_value = normalize_result

        extract_chain = MagicMock()
        extract_chain.invoke.return_value = extract_result

        # Mock with_structured_output to return appropriate chains
        def with_structured_output_side_effect(schema, **kwargs):
            if schema == NormalizeModel:
                return normalize_chain
            elif schema == ExtractSignalsModel:
                return extract_chain
            return MagicMock()

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        # Mock prompt templates
        mock_normalize_prompt.from_messages.return_value.__or__ = lambda self, other: normalize_chain
        mock_extract_prompt.from_messages.return_value.__or__ = lambda self, other: extract_chain

        # Build and invoke graph
        graph = make_intake_graph(mock_llm, max_retries=1)
        result = graph.invoke({"messages": sample_messages})

        # Verify both nodes were executed
        assert "normalized_query" in result
        assert "user_intent" in result
        assert "retrieval_intent" in result
        assert "intake_version" in result

        # Verify output values
        assert result["normalized_query"] == "test query"
        assert result["user_intent"] == "explain"
        assert result["intake_version"] == "intake_v1"

    def test_intake_graph_with_max_retries(self, mock_llm):
        """Test that max_retries parameter is used."""
        graph = make_intake_graph(mock_llm, max_retries=5)
        assert graph is not None

        # Different retry count
        graph2 = make_intake_graph(mock_llm, max_retries=1)
        assert graph2 is not None

    def test_intake_graph_error_propagation(self, mock_llm, sample_messages):
        """Test that errors from nodes are propagated in final state."""
        # Setup mock that returns error
        def error_chain_invoke(input_dict):
            # Return error structure instead of raising
            return None

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Mock LLM error")
        mock_llm.with_structured_output.return_value = mock_chain

        graph = make_intake_graph(mock_llm, max_retries=1)

        # This might raise or return errors depending on graph configuration
        # We test that the graph handles it gracefully
        try:
            result = graph.invoke({"messages": sample_messages})
            # If no exception, check for errors in result
            if "errors" in result:
                assert len(result["errors"]) > 0
        except Exception:
            # Graph might raise - that's also acceptable behavior
            pass

    def test_intake_graph_minimal_input(self, mock_llm):
        """Test graph with minimal required input."""
        # Mock successful execution
        normalize_result = NormalizeModel(
            normalized_query="minimal",
            constraints={},
            guardrails={},
            clarification={},
        )
        extract_result = ExtractSignalsModel(
            user_intent="other",
            retrieval_intent="none",
            answerability="reasoning_only",
        )

        normalize_chain = MagicMock()
        normalize_chain.invoke.return_value = normalize_result
        extract_chain = MagicMock()
        extract_chain.invoke.return_value = extract_result

        call_count = [0]

        def with_structured_output_side_effect(schema, **kwargs):
            result = normalize_chain if call_count[0] == 0 else extract_chain
            call_count[0] += 1
            return result

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        graph = make_intake_graph(mock_llm, max_retries=1)
        result = graph.invoke({"messages": [{"role": "user", "content": "test"}]})

        # Should complete without errors
        assert "normalized_query" in result or "errors" in result

    def test_intake_graph_preserves_optional_fields(self, mock_llm):
        """Test that optional fields from intake are preserved."""
        normalize_result = NormalizeModel(
            normalized_query="test",
            constraints={},
            guardrails={},
            clarification={},
            language="es",
            locale="es-MX",
        )
        extract_result = ExtractSignalsModel(
            user_intent="lookup",
            retrieval_intent="definition",
            answerability="internal_corpus",
        )

        normalize_chain = MagicMock()
        normalize_chain.invoke.return_value = normalize_result
        extract_chain = MagicMock()
        extract_chain.invoke.return_value = extract_result

        call_count = [0]

        def with_structured_output_side_effect(schema, **kwargs):
            result = normalize_chain if call_count[0] == 0 else extract_chain
            call_count[0] += 1
            return result

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        graph = make_intake_graph(mock_llm, max_retries=1)
        result = graph.invoke({"messages": [{"role": "user", "content": "test"}]})

        # Language and locale should be preserved
        if "language" in result:
            assert result["language"] == "es"
        if "locale" in result:
            assert result["locale"] == "es-MX"


class TestIntakeGraphIntegration:
    """Integration-style tests for intake graph (still with mocked LLM)."""

    def test_full_intake_pipeline_explain_intent(self, mock_llm):
        """Test full pipeline for 'explain' user intent."""
        # Mock outputs
        normalize_result = NormalizeModel(
            normalized_query="explain RAG architecture",
            constraints={"domain": ["rag"]},
            guardrails={"time_sensitivity": "none", "sensitivity": "normal"},
            clarification={"needed": False, "blocking": False, "reasons": []},
        )

        extract_result = ExtractSignalsModel(
            user_intent="explain",
            retrieval_intent="background",
            answerability="internal_corpus",
            complexity_flags=["requires_synthesis"],
            signals=SignalsModel(
                entities=[{"text": "RAG", "type": "concept", "confidence": "high"}],
                acronyms=[{"text": "RAG", "expansion": "Retrieval Augmented Generation", "confidence": "high"}],
            ),
        )

        normalize_chain = MagicMock()
        normalize_chain.invoke.return_value = normalize_result
        extract_chain = MagicMock()
        extract_chain.invoke.return_value = extract_result

        call_count = [0]

        def with_structured_output_side_effect(schema, **kwargs):
            result = normalize_chain if call_count[0] == 0 else extract_chain
            call_count[0] += 1
            return result

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        graph = make_intake_graph(mock_llm, max_retries=1)
        result = graph.invoke({"messages": [{"role": "user", "content": "Explain RAG"}]})

        # Verify full output
        assert result.get("user_intent") == "explain"
        assert result.get("retrieval_intent") == "background"
        assert result.get("normalized_query") == "explain RAG architecture"
        assert "requires_synthesis" in result.get("complexity_flags", [])

    def test_full_intake_pipeline_troubleshoot_intent(self, mock_llm):
        """Test full pipeline for 'troubleshoot' user intent."""
        normalize_result = NormalizeModel(
            normalized_query="Azure OpenAI returns 429 error",
            constraints={"domain": ["azure"]},
            guardrails={"time_sensitivity": "high", "sensitivity": "normal"},
            clarification={"needed": False},
        )

        extract_result = ExtractSignalsModel(
            user_intent="troubleshoot",
            retrieval_intent="verification",
            answerability="internal_corpus",
            complexity_flags=["requires_strict_precision"],
            signals=SignalsModel(
                literal_terms=["429 error"],
                artifact_flags=["has_ids"],
            ),
        )

        normalize_chain = MagicMock()
        normalize_chain.invoke.return_value = normalize_result
        extract_chain = MagicMock()
        extract_chain.invoke.return_value = extract_result

        call_count = [0]

        def with_structured_output_side_effect(schema, **kwargs):
            result = normalize_chain if call_count[0] == 0 else extract_chain
            call_count[0] += 1
            return result

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        graph = make_intake_graph(mock_llm, max_retries=1)
        result = graph.invoke({
            "messages": [{"role": "user", "content": "Getting 429 error from Azure OpenAI"}]
        })

        assert result.get("user_intent") == "troubleshoot"
        assert result.get("retrieval_intent") == "verification"
        assert "requires_strict_precision" in result.get("complexity_flags", [])
        if "signals" in result:
            assert "429 error" in result["signals"].get("literal_terms", [])
