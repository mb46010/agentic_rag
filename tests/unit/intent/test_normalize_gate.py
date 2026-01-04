# tests/unit/intent/test_normalize_gate.py
"""Unit tests for normalize_gate node."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_rag.intent.nodes.normalize_gate import NormalizeModel, make_normalize_gate_node


class TestNormalizeGateNode:
    """Tests for normalize_gate node."""

    def test_make_normalize_gate_node_returns_callable(self, mock_llm):
        """Test that make_normalize_gate_node returns a callable."""
        node = make_normalize_gate_node(mock_llm)
        assert callable(node)

    def test_normalize_gate_missing_messages(self, mock_llm):
        """Test error handling when messages are missing."""
        node = make_normalize_gate_node(mock_llm)
        state = {}

        result = node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "normalize_gate"
        assert error["type"] == "schema_validation"
        assert "messages" in error["message"].lower()
        assert error["retryable"] is False

    def test_normalize_gate_empty_messages(self, mock_llm):
        """Test error handling when messages list is empty."""
        node = make_normalize_gate_node(mock_llm)
        state = {"messages": []}

        result = node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0]["node"] == "normalize_gate"

    def test_normalize_gate_invalid_messages_type(self, mock_llm):
        """Test error handling when messages is not a list."""
        node = make_normalize_gate_node(mock_llm)
        state = {"messages": "not a list"}

        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["type"] == "schema_validation"

    def test_normalize_gate_successful_invocation(self, mock_llm, sample_state, mock_normalize_output):
        """Test successful normalize_gate invocation."""
        # Setup mock to return valid NormalizeModel
        mock_result = NormalizeModel(
            normalized_query=mock_normalize_output["normalized_query"],
            constraints=mock_normalize_output["constraints"],
            guardrails=mock_normalize_output["guardrails"],
            clarification=mock_normalize_output["clarification"],
            language=mock_normalize_output["language"],
            locale=mock_normalize_output["locale"],
        )

        # Mock the chain behavior on mock_llm
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        node = make_normalize_gate_node(mock_llm)
        result = node(sample_state)

        # Verify structure
        assert "normalized_query" in result
        assert "constraints" in result
        assert "guardrails" in result
        assert "clarification" in result
        assert "intake_version" in result

        # Verify values
        assert result["normalized_query"] == mock_normalize_output["normalized_query"]
        assert result["intake_version"] == "intake_v1"
        assert result["language"] == "en"
        assert result["locale"] == "en-US"

        # Verify chain was called
        assert mock_llm.with_structured_output.return_value.invoke.called

    def test_normalize_gate_validation_error(self, mock_llm, sample_state):
        """Test error handling when LLM output fails validation."""
        # Mock chain that raises ValidationError
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("normalized_query",), "msg": "Field required"}]
        )
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_normalize_gate_node(mock_llm)
        result = node(sample_state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "normalize_gate"
        assert error["type"] == "model_output_parse"
        assert error["retryable"] is True
        assert "details" in error
        assert "validation_errors" in error["details"]

    def test_normalize_gate_runtime_error(self, mock_llm, sample_state):
        """Test error handling for runtime errors."""
        # Mock chain that raises generic exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Network error")
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_normalize_gate_node(mock_llm)
        result = node(sample_state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "normalize_gate"
        assert error["type"] == "runtime_error"
        assert error["retryable"] is True
        assert "Network error" in error["message"]

    def test_normalize_gate_optional_fields_omitted(self, mock_llm, sample_state):
        """Test that optional fields (language, locale) are only included when present."""
        # Setup mock without language/locale
        mock_result = NormalizeModel(
            normalized_query="test query",
            constraints={},
            guardrails={},
            clarification={},
            language=None,
            locale=None,
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_normalize_gate_node(mock_llm)
        result = node(sample_state)

        # Language and locale should not be in result when None
        assert "language" not in result
        assert "locale" not in result
        # But required fields should be present
        assert "normalized_query" in result
        assert "intake_version" in result

    def test_normalize_model_validation(self):
        """Test NormalizeModel Pydantic validation."""
        # Valid model
        valid = NormalizeModel(
            normalized_query="test",
            constraints={},
            guardrails={},
            clarification={},
        )
        assert valid.normalized_query == "test"

        # Invalid: missing required field
        with pytest.raises(ValidationError):
            NormalizeModel(
                constraints={},
                guardrails={},
                clarification={},
            )

        # Invalid: empty normalized_query
        with pytest.raises(ValidationError):
            NormalizeModel(
                normalized_query="",
                constraints={},
                guardrails={},
                clarification={},
            )
