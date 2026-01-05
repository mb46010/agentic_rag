# tests/unit/planner/test_planner_node.py
"""Unit tests for planner node."""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from agentic_rag.planner.nodes.planner import make_planner_node
from agentic_rag.planner.state import (
    PlannerState,
    ClarifyingQuestion,
    RetrievalRound,
    RetrievalModeSpec,
    LiteralConstraints,
)


class TestPlannerNode:
    """Tests for planner node logic."""

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_direct_answer_strategy(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node returns direct_answer strategy."""
        planner_result = PlannerState(
            goal="What is 2+2?",
            strategy="direct_answer",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        assert "plan" in result
        assert result["plan"]["strategy"] == "direct_answer"
        assert result["plan"]["retrieval_rounds"] == []

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_retrieve_then_answer_strategy(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node returns retrieve_then_answer strategy."""
        planner_result = PlannerState(
            goal="Configure Azure OpenAI",
            strategy="retrieve_then_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["Azure OpenAI setup"],
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        assert "plan" in result
        assert result["plan"]["strategy"] == "retrieve_then_answer"
        assert len(result["plan"]["retrieval_rounds"]) == 1

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_clarify_then_retrieve_strategy(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node returns clarify_then_retrieve strategy."""
        planner_result = PlannerState(
            goal="Configure library",
            strategy="clarify_then_retrieve",
            clarifying_questions=[
                ClarifyingQuestion(
                    question="Which version?",
                    reason="missing_version",
                    blocking=True,
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        assert "plan" in result
        assert result["plan"]["strategy"] == "clarify_then_retrieve"
        assert len(result["plan"]["clarifying_questions"]) == 1

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_defer_or_refuse_strategy(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node returns defer_or_refuse strategy."""
        planner_result = PlannerState(
            goal="Restricted request",
            strategy="defer_or_refuse",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        assert "plan" in result
        assert result["plan"]["strategy"] == "defer_or_refuse"

    def test_planner_node_missing_messages(self, mock_llm):
        """Test that planner node returns error for missing messages."""
        node = make_planner_node(mock_llm)

        # Call with missing messages
        result = node({})

        assert "errors" in result
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "planner"
        assert result["errors"][0]["type"] == "schema_validation"
        assert result["errors"][0]["retryable"] is False

    def test_planner_node_empty_messages(self, mock_llm):
        """Test that planner node returns error for empty messages list."""
        node = make_planner_node(mock_llm)

        # Call with empty messages
        result = node({"messages": []})

        assert "errors" in result
        assert result["errors"][0]["node"] == "planner"
        assert result["errors"][0]["type"] == "schema_validation"

    def test_planner_node_invalid_messages_type(self, mock_llm):
        """Test that planner node returns error for invalid messages type."""
        node = make_planner_node(mock_llm)

        # Call with non-list messages
        result = node({"messages": "not a list"})

        assert "errors" in result
        assert result["errors"][0]["node"] == "planner"
        assert result["errors"][0]["type"] == "schema_validation"

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_validation_error(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node handles validation error from LLM output."""
        # Mock chain that returns invalid output
        mock_chain = MagicMock()
        # Return a dict that will fail PlannerState validation
        mock_chain.invoke.return_value = {
            "goal": "x",  # Too short (< 5 chars)
            "strategy": "direct_answer",
        }

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Should return error
        assert "errors" in result
        assert result["errors"][0]["node"] == "planner"
        assert result["errors"][0]["type"] == "model_output_parse"
        assert result["errors"][0]["retryable"] is True

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_runtime_error(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node handles runtime error."""
        # Mock chain that raises exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM API error")

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Should return error
        assert "errors" in result
        assert result["errors"][0]["node"] == "planner"
        assert result["errors"][0]["type"] == "runtime_error"
        assert result["errors"][0]["retryable"] is True
        assert "LLM API error" in result["errors"][0]["message"]

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_enforces_blocking_clarification_invariant(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test that planner enforces blocking clarification invariant."""
        # Return a plan with blocking clarifications but wrong strategy
        planner_result = PlannerState(
            goal="Configure library",
            strategy="retrieve_then_answer",  # Wrong strategy
            clarifying_questions=[
                ClarifyingQuestion(
                    question="Which version?",
                    reason="missing_version",
                    blocking=True,
                )
            ],
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["query"],
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Should force strategy to clarify_then_retrieve
        assert result["plan"]["strategy"] == "clarify_then_retrieve"
        # Should clear retrieval_rounds
        assert result["plan"]["retrieval_rounds"] == []

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_enforces_strategy_invariant(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test that planner clears retrieval_rounds for non-retrieve strategies."""
        # Return a plan with direct_answer but has retrieval rounds (invalid)
        planner_result = PlannerState(
            goal="What is 2+2?",
            strategy="direct_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["query"],
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Should clear retrieval_rounds for direct_answer
        assert result["plan"]["strategy"] == "direct_answer"
        assert result["plan"]["retrieval_rounds"] == []

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_includes_all_intake_fields(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test that planner node receives all intake fields."""
        planner_result = PlannerState(
            goal="test goal",
            strategy="direct_answer",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Verify chain was invoked with correct payload
        invoke_call = mock_chain.invoke.call_args
        payload = invoke_call[0][0]

        # Check all expected fields are in payload
        assert "messages" in payload
        assert "normalized_query" in payload
        assert "constraints" in payload
        assert "guardrails" in payload
        assert "clarification" in payload
        assert "user_intent" in payload
        assert "retrieval_intent" in payload
        assert "answerability" in payload
        assert "complexity_flags" in payload
        assert "signals" in payload

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_handles_missing_optional_fields(
        self, mock_prompt, mock_llm
    ):
        """Test that planner node handles missing optional fields gracefully."""
        minimal_state = {
            "messages": [{"role": "user", "content": "test"}],
        }

        planner_result = PlannerState(
            goal="test goal",
            strategy="direct_answer",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(minimal_state)

        # Should complete without error
        assert "plan" in result or "errors" not in result

        # Verify chain was invoked with defaults for missing fields
        invoke_call = mock_chain.invoke.call_args
        payload = invoke_call[0][0]

        assert payload["normalized_query"] == ""
        assert payload["constraints"] == {}
        assert payload["guardrails"] == {}
        assert payload["clarification"] == {}

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_with_literal_constraints(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node with literal constraints."""
        # Add literal terms to intake state
        intake_with_literals = {
            **sample_intake_state,
            "signals": {
                "literal_terms": ["ERROR-123", "config.yaml"],
                "artifact_flags": ["has_stacktrace"],
            },
        }

        planner_result = PlannerState(
            goal="Debug error",
            strategy="retrieve_then_answer",
            literal_constraints=LiteralConstraints(
                must_preserve_terms=["ERROR-123"],
                must_match_exactly=True,
            ),
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="precision",
                    query_variants=["ERROR-123 troubleshooting"],
                    retrieval_modes=[
                        RetrievalModeSpec(type="bm25", k=30)
                    ],
                    use_hyde=False,
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(intake_with_literals)

        assert result["plan"]["literal_constraints"]["must_preserve_terms"] == [
            "ERROR-123"
        ]
        assert result["plan"]["literal_constraints"]["must_match_exactly"] is True
        # Should not use HyDE with literal constraints
        assert result["plan"]["retrieval_rounds"][0]["use_hyde"] is False

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_with_multiple_retrieval_rounds(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test planner node with multiple retrieval rounds."""
        planner_result = PlannerState(
            goal="Complex query",
            strategy="retrieve_then_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["broad query"],
                ),
                RetrievalRound(
                    round_id=1,
                    purpose="precision",
                    query_variants=["specific query"],
                ),
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        assert len(result["plan"]["retrieval_rounds"]) == 2
        assert result["plan"]["retrieval_rounds"][0]["purpose"] == "recall"
        assert result["plan"]["retrieval_rounds"][1]["purpose"] == "precision"

    @patch("agentic_rag.planner.nodes.planner.ChatPromptTemplate")
    def test_planner_node_model_dump_serialization(
        self, mock_prompt, mock_llm, sample_intake_state
    ):
        """Test that planner node output is properly serialized."""
        planner_result = PlannerState(
            goal="test goal",
            strategy="retrieve_then_answer",
            retrieval_rounds=[
                RetrievalRound(
                    round_id=0,
                    purpose="recall",
                    query_variants=["query"],
                    retrieval_modes=[
                        RetrievalModeSpec(type="hybrid", k=20, alpha=0.5)
                    ],
                )
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = planner_result

        mock_llm.with_structured_output.return_value = mock_chain
        mock_prompt.from_messages.return_value.__or__ = lambda self, other: mock_chain

        node = make_planner_node(mock_llm)
        result = node(sample_intake_state)

        # Plan should be a dict (serialized)
        assert isinstance(result["plan"], dict)
        assert result["plan"]["goal"] == "test goal"
        assert result["plan"]["strategy"] == "retrieve_then_answer"

        # Nested objects should also be serialized
        assert isinstance(result["plan"]["retrieval_rounds"], list)
        assert isinstance(result["plan"]["retrieval_rounds"][0], dict)
