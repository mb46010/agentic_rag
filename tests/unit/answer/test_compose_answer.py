# tests/unit/answer/test_compose_answer.py
"""Unit tests for compose_answer node."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agentic_rag.answer.nodes.compose_answer import make_compose_answer_node
from agentic_rag.answer.state import ComposeAnswerModel, Citation


class TestComposeAnswer:
    """Tests for compose_answer node."""

    def test_successful_answer_composition(self, mock_llm, sample_answer_state):
        """Should successfully compose an answer."""
        # Mock the LLM response
        mock_output = ComposeAnswerModel(
            final_answer="LangGraph is a library for building stateful applications with LLMs.",
            citations=[
                Citation(
                    evidence_id="ev_001",
                    text="LangGraph is a library for building stateful applications.",
                    source="docs.langgraph.com",
                )
            ],
            followups=["How do I install LangGraph?", "What are the main features?"],
            used_evidence_ids=["ev_001", "ev_002"],
            asked_clarification=False,
            refusal=False,
        )

        mock_llm._mock_chain.invoke.return_value = mock_output

        state = {**sample_answer_state, "answer_mode": "answer"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert result["final_answer"] == mock_output.final_answer
        assert len(result["citations"]) == 1
        assert result["citations"][0]["evidence_id"] == "ev_001"
        assert result["followups"] == mock_output.followups
        assert result["answer_meta"]["used_evidence_ids"] == ["ev_001", "ev_002"]
        assert result["answer_meta"]["asked_clarification"] is False
        assert result["answer_meta"]["refusal"] is False

        # Verify chain was called
        mock_llm._mock_chain.invoke.assert_called_once()

    def test_handles_missing_messages(self, mock_llm, sample_answer_state):
        """Should return error when messages are missing."""
        state = {**sample_answer_state, "messages": None}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0]["node"] == "compose_answer"
        assert result["errors"][0]["type"] == "schema_validation"
        assert "messages" in result["errors"][0]["message"].lower()

    def test_handles_empty_messages(self, mock_llm, sample_answer_state):
        """Should return error when messages are empty."""
        state = {**sample_answer_state, "messages": []}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["type"] == "schema_validation"

    def test_handles_invalid_messages_type(self, mock_llm, sample_answer_state):
        """Should return error when messages is not a list."""
        state = {**sample_answer_state, "messages": "invalid"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["type"] == "schema_validation"

    def test_handles_validation_error(self, mock_llm, sample_answer_state):
        """Should handle validation error from LLM output."""
        # Mock chain to return invalid output that will fail validation
        invalid_output = {"final_answer": 123}  # Invalid type
        mock_llm._mock_chain.invoke.return_value = invalid_output

        state = {**sample_answer_state, "answer_mode": "answer"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["node"] == "compose_answer"
        assert result["errors"][0]["type"] == "model_output_parse"
        assert result["errors"][0]["retryable"] is True

    def test_handles_runtime_error(self, mock_llm, sample_answer_state):
        """Should handle runtime errors from LLM."""
        # Mock chain to raise an exception
        mock_llm._mock_chain.invoke.side_effect = RuntimeError("API timeout")

        state = {**sample_answer_state, "answer_mode": "answer"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["node"] == "compose_answer"
        assert result["errors"][0]["type"] == "runtime_error"
        assert "API timeout" in result["errors"][0]["message"]
        assert result["errors"][0]["retryable"] is True

    def test_passes_answer_mode_to_llm(self, mock_llm, sample_answer_state):
        """Should pass answer_mode to LLM payload."""
        mock_output = ComposeAnswerModel(
            final_answer="Clarifying question",
            citations=[],
            followups=[],
            used_evidence_ids=[],
            asked_clarification=True,
            refusal=False,
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        state = {**sample_answer_state, "answer_mode": "clarify"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        # Verify the payload includes answer_mode
        call_args = mock_llm._mock_chain.invoke.call_args[0][0]
        assert call_args["answer_mode"] == "clarify"

    def test_coerces_evidence_from_state(self, mock_llm, sample_answer_state):
        """Should coerce evidence items from state."""
        mock_output = ComposeAnswerModel(
            final_answer="Answer",
            citations=[],
            followups=[],
            used_evidence_ids=[],
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        # Include mixed valid and invalid evidence
        state = {
            **sample_answer_state,
            "final_evidence": [
                {"evidence_id": "ev_001", "text": "Valid evidence"},
                "invalid_evidence",  # Should be skipped
                {"evidence_id": "ev_002", "text": "Another valid"},
            ],
        }

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        # Verify evidence was coerced
        call_args = mock_llm._mock_chain.invoke.call_args[0][0]
        assert len(call_args["final_evidence"]) == 2

    def test_coerces_coverage_from_state(self, mock_llm, sample_answer_state):
        """Should coerce coverage from state."""
        mock_output = ComposeAnswerModel(
            final_answer="Answer",
            citations=[],
            followups=[],
            used_evidence_ids=[],
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        state = {
            **sample_answer_state,
            "coverage": {
                "confidence": 0.9,
                "covered": ["item1"],
                "missing": [],
                "contradictions": [],
            },
        }

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        # Verify coverage was passed
        call_args = mock_llm._mock_chain.invoke.call_args[0][0]
        assert call_args["coverage"]["confidence"] == 0.9

    def test_handles_invalid_coverage(self, mock_llm, sample_answer_state):
        """Should handle invalid coverage gracefully."""
        mock_output = ComposeAnswerModel(
            final_answer="Answer",
            citations=[],
            followups=[],
            used_evidence_ids=[],
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        state = {**sample_answer_state, "coverage": "invalid"}

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        # Should use default coverage
        call_args = mock_llm._mock_chain.invoke.call_args[0][0]
        assert call_args["coverage"]["confidence"] == 0.0

    def test_merges_answer_meta(self, mock_llm, sample_answer_state):
        """Should merge answer_meta from state with new values."""
        mock_output = ComposeAnswerModel(
            final_answer="Answer",
            citations=[],
            followups=[],
            used_evidence_ids=["ev_001"],
            asked_clarification=False,
            refusal=False,
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        state = {
            **sample_answer_state,
            "answer_meta": {
                "answer_version": "answer_v1",
                "mode": "answer",
                "coverage_confidence": 0.8,
            },
        }

        node = make_compose_answer_node(mock_llm)
        result = node(state)

        # Should preserve existing meta and add new values
        assert result["answer_meta"]["answer_version"] == "answer_v1"
        assert result["answer_meta"]["mode"] == "answer"
        assert result["answer_meta"]["coverage_confidence"] == 0.8
        assert result["answer_meta"]["used_evidence_ids"] == ["ev_001"]

    def test_includes_all_required_fields_in_payload(self, mock_llm, sample_answer_state):
        """Should include all required fields in LLM payload."""
        mock_output = ComposeAnswerModel(
            final_answer="Answer",
            citations=[],
            followups=[],
            used_evidence_ids=[],
        )
        mock_llm._mock_chain.invoke.return_value = mock_output

        node = make_compose_answer_node(mock_llm)
        result = node(sample_answer_state)

        call_args = mock_llm._mock_chain.invoke.call_args[0][0]

        # Verify all required fields are present
        assert "messages" in call_args
        assert "answer_mode" in call_args
        assert "plan" in call_args
        assert "constraints" in call_args
        assert "guardrails" in call_args
        assert "normalized_query" in call_args
        assert "final_evidence" in call_args
        assert "coverage" in call_args
        assert "language" in call_args
        assert "locale" in call_args
