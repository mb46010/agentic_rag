# tests/unit/intent/test_extract_signals.py
"""Unit tests for extract_signals node."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_rag.intent.nodes.extract_signals import (
    AcronymModel,
    EntityModel,
    ExtractSignalsModel,
    SignalsModel,
    make_extract_signals_node,
)


class TestExtractSignalsNode:
    """Tests for extract_signals node."""

    def test_make_extract_signals_node_returns_callable(self, mock_llm):
        """Test that make_extract_signals_node returns a callable."""
        node = make_extract_signals_node(mock_llm)
        assert callable(node)

    def test_extract_signals_missing_messages(self, mock_llm):
        """Test error handling when messages are missing."""
        node = make_extract_signals_node(mock_llm)
        state = {}

        result = node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "extract_signals"
        assert error["type"] == "schema_validation"
        assert "messages" in error["message"].lower()

    def test_extract_signals_empty_messages(self, mock_llm):
        """Test error handling when messages list is empty."""
        node = make_extract_signals_node(mock_llm)
        state = {"messages": []}

        result = node(state)

        assert "errors" in result
        assert result["errors"][0]["node"] == "extract_signals"

    def test_extract_signals_successful_invocation(
        self, mock_llm, sample_state, mock_normalize_output, mock_extract_signals_output
    ):
        """Test successful extract_signals invocation."""
        # Combine states
        full_state = {**sample_state, **mock_normalize_output}

        # Setup mock to return valid ExtractSignalsModel
        mock_result = ExtractSignalsModel(
            user_intent=mock_extract_signals_output["user_intent"],
            retrieval_intent=mock_extract_signals_output["retrieval_intent"],
            answerability=mock_extract_signals_output["answerability"],
            complexity_flags=mock_extract_signals_output["complexity_flags"],
            signals=SignalsModel(**mock_extract_signals_output["signals"]),
        )

        # Mock the chain behavior on mock_llm
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        node = make_extract_signals_node(mock_llm)
        result = node(full_state)

        # Verify structure
        assert "user_intent" in result
        assert "retrieval_intent" in result
        assert "answerability" in result
        assert "complexity_flags" in result
        assert "signals" in result

        # Verify values
        assert result["user_intent"] == "plan"
        assert result["retrieval_intent"] == "procedure"
        assert result["answerability"] == "internal_corpus"
        assert "requires_synthesis" in result["complexity_flags"]
        assert isinstance(result["signals"], dict)

        # Verify chain was called with context
        call_args = mock_llm.with_structured_output.return_value.invoke.call_args
        assert call_args is not None
        invoke_input = call_args[0][0]
        # If it's a PromptValue, we can convert to string to check content
        input_str = str(invoke_input)
        assert "normalized_query" in input_str
        assert "azure" in input_str.lower()

    def test_extract_signals_validation_error(self, mock_llm, sample_state, mock_normalize_output):
        """Test error handling when LLM output fails validation."""
        full_state = {**sample_state, **mock_normalize_output}

        # Mock chain that raises ValidationError
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("user_intent",), "msg": "Field required"}]
        )
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_extract_signals_node(mock_llm)
        result = node(full_state)

        assert "errors" in result
        error = result["errors"][0]
        assert error["node"] == "extract_signals"
        assert error["type"] == "model_output_parse"
        assert error["retryable"] is True

    def test_extract_signals_runtime_error(self, mock_llm, sample_state, mock_normalize_output):
        """Test error handling for runtime errors."""
        full_state = {**sample_state, **mock_normalize_output}

        # Mock chain that raises generic exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM API timeout")
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_extract_signals_node(mock_llm)
        result = node(full_state)

        assert "errors" in result
        error = result["errors"][0]
        assert error["node"] == "extract_signals"
        assert error["type"] == "runtime_error"
        assert "timeout" in error["message"].lower()

    def test_extract_signals_with_defaults(self, mock_llm, sample_state):
        """Test extract_signals with missing normalized fields (uses defaults)."""
        # State with only messages
        state = {"messages": sample_state["messages"]}

        # Setup minimal mock result
        mock_result = ExtractSignalsModel(
            user_intent="other",
            retrieval_intent="none",
            answerability="reasoning_only",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_extract_signals_node(mock_llm)
        result = node(state)

        # Should use conservative defaults
        assert result["user_intent"] == "other"
        assert result["complexity_flags"] == []
        assert result["signals"]["entities"] == []

    def test_signals_model_with_entities(self):
        """Test SignalsModel with entities."""
        signals = SignalsModel(
            entities=[
                EntityModel(text="Azure", type="product", confidence="high"),
                EntityModel(text="Python", type="component", confidence="medium"),
            ]
        )
        assert len(signals.entities) == 2
        assert signals.entities[0].text == "Azure"

    def test_signals_model_with_acronyms(self):
        """Test SignalsModel with acronyms."""
        signals = SignalsModel(
            acronyms=[
                AcronymModel(text="API", expansion="Application Programming Interface", confidence="high"),
                AcronymModel(text="RAG", expansion=None, confidence="low"),
            ]
        )
        assert len(signals.acronyms) == 2
        assert signals.acronyms[0].expansion is not None
        assert signals.acronyms[1].expansion is None

    def test_signals_model_forbid_extra(self):
        """Test that SignalsModel forbids extra fields."""
        with pytest.raises(ValidationError):
            SignalsModel(
                entities=[],
                extra_field="not allowed",
            )

    def test_entity_model_validation(self):
        """Test EntityModel Pydantic validation."""
        # Valid entity
        entity = EntityModel(text="test", type="product", confidence="high")
        assert entity.text == "test"

        # Invalid: wrong type literal
        with pytest.raises(ValidationError):
            EntityModel(text="test", type="invalid_type", confidence="high")

        # Invalid: wrong confidence literal
        with pytest.raises(ValidationError):
            EntityModel(text="test", type="product", confidence="very_high")

    def test_extract_signals_model_validation(self):
        """Test ExtractSignalsModel validation."""
        # Valid model
        valid = ExtractSignalsModel(
            user_intent="explain",
            retrieval_intent="definition",
            answerability="internal_corpus",
        )
        assert valid.user_intent == "explain"
        assert valid.complexity_flags == []  # default

        # Invalid: wrong user_intent literal
        with pytest.raises(ValidationError):
            ExtractSignalsModel(
                user_intent="invalid_intent",
                retrieval_intent="definition",
                answerability="internal_corpus",
            )

    def test_extract_signals_preserves_literal_terms(self, mock_llm, sample_state):
        """Test that literal terms are preserved in signals."""
        mock_result = ExtractSignalsModel(
            user_intent="troubleshoot",
            retrieval_intent="verification",
            answerability="internal_corpus",
            signals=SignalsModel(literal_terms=["ERROR-12345", "/var/log/app.log", "failed to connect"]),
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_extract_signals_node(mock_llm)
        result = node(sample_state)

        assert "signals" in result
        assert "literal_terms" in result["signals"]
        assert "ERROR-12345" in result["signals"]["literal_terms"]

    def test_extract_signals_complexity_flags(self, mock_llm, sample_state):
        """Test complexity flags extraction."""
        mock_result = ExtractSignalsModel(
            user_intent="plan",
            retrieval_intent="mixed",
            answerability="mixed",
            complexity_flags=["multi_intent", "multi_domain", "requires_synthesis"],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm.with_structured_output.side_effect = None
        mock_llm.with_structured_output.return_value = mock_chain

        node = make_extract_signals_node(mock_llm)
        result = node(sample_state)

        assert len(result["complexity_flags"]) == 3
        assert "multi_intent" in result["complexity_flags"]
        assert "requires_synthesis" in result["complexity_flags"]
