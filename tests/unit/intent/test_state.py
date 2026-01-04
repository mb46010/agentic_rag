# tests/unit/intent/test_state.py
"""Unit tests for intent state models and utilities."""

import pytest

from agentic_rag.intent.state import (
    add_errors,
    Constraints,
    Guardrails,
    Signals,
    Entity,
    Acronym,
    Clarification,
    IntakeError,
)


class TestAddErrors:
    """Tests for add_errors reducer function."""

    def test_add_errors_both_none(self):
        """Test add_errors when both inputs are None."""
        result = add_errors(None, None)
        assert result == []

    def test_add_errors_existing_none(self):
        """Test add_errors when existing is None."""
        new_errors = [{"node": "test", "type": "error"}]
        result = add_errors(None, new_errors)
        assert result == new_errors

    def test_add_errors_new_none(self):
        """Test add_errors when new is None."""
        existing_errors = [{"node": "test1", "type": "error1"}]
        result = add_errors(existing_errors, None)
        assert result == existing_errors

    def test_add_errors_both_present(self):
        """Test add_errors when both have errors."""
        existing = [{"node": "test1", "type": "error1"}]
        new = [{"node": "test2", "type": "error2"}]
        result = add_errors(existing, new)
        assert len(result) == 2
        assert result[0]["node"] == "test1"
        assert result[1]["node"] == "test2"

    def test_add_errors_preserves_order(self):
        """Test that add_errors preserves error order."""
        existing = [{"id": 1}, {"id": 2}]
        new = [{"id": 3}, {"id": 4}]
        result = add_errors(existing, new)
        assert [e["id"] for e in result] == [1, 2, 3, 4]


class TestConstraints:
    """Tests for Constraints TypedDict."""

    def test_constraints_empty(self):
        """Test creating empty Constraints."""
        c: Constraints = {}
        assert c == {}

    def test_constraints_with_domain(self):
        """Test Constraints with domain."""
        c: Constraints = {"domain": ["azure", "kubernetes"]}
        assert "azure" in c["domain"]

    def test_constraints_with_format(self):
        """Test Constraints with format."""
        c: Constraints = {"format": ["no_code", "bullet_points"]}
        assert len(c["format"]) == 2

    def test_constraints_partial(self):
        """Test partial Constraints (total=False allows this)."""
        c: Constraints = {"domain": ["rag"]}
        assert "domain" in c
        assert "format" not in c


class TestGuardrails:
    """Tests for Guardrails TypedDict."""

    def test_guardrails_full(self):
        """Test complete Guardrails."""
        g: Guardrails = {
            "time_sensitivity": "high",
            "context_dependency": "strong",
            "sensitivity": "elevated",
            "pii_present": True,
        }
        assert g["time_sensitivity"] == "high"
        assert g["pii_present"] is True

    def test_guardrails_partial(self):
        """Test partial Guardrails."""
        g: Guardrails = {"sensitivity": "normal"}
        assert g["sensitivity"] == "normal"


class TestSignals:
    """Tests for Signals TypedDict."""

    def test_signals_empty(self):
        """Test empty Signals."""
        s: Signals = {}
        assert s == {}

    def test_signals_with_entities(self):
        """Test Signals with entities."""
        s: Signals = {
            "entities": [
                {"text": "Azure", "type": "product", "confidence": "high"}
            ]
        }
        assert len(s["entities"]) == 1

    def test_signals_with_acronyms(self):
        """Test Signals with acronyms."""
        s: Signals = {
            "acronyms": [
                {"text": "API", "expansion": "Application Programming Interface", "confidence": "high"},
                {"text": "RAG", "expansion": None, "confidence": "low"},
            ]
        }
        assert len(s["acronyms"]) == 2
        assert s["acronyms"][1]["expansion"] is None

    def test_signals_with_literal_terms(self):
        """Test Signals with literal terms."""
        s: Signals = {
            "literal_terms": ["ERROR-123", "/var/log/app.log"]
        }
        assert "ERROR-123" in s["literal_terms"]

    def test_signals_with_artifact_flags(self):
        """Test Signals with artifact flags."""
        s: Signals = {
            "artifact_flags": ["has_code", "has_stacktrace"]
        }
        assert "has_code" in s["artifact_flags"]


class TestEntity:
    """Tests for Entity TypedDict."""

    def test_entity_full(self):
        """Test complete Entity."""
        e: Entity = {
            "text": "Azure OpenAI",
            "type": "product",
            "confidence": "high",
        }
        assert e["text"] == "Azure OpenAI"
        assert e["type"] == "product"

    def test_entity_partial(self):
        """Test partial Entity."""
        e: Entity = {"text": "test"}
        assert "text" in e


class TestAcronym:
    """Tests for Acronym TypedDict."""

    def test_acronym_with_expansion(self):
        """Test Acronym with expansion."""
        a: Acronym = {
            "text": "RAG",
            "expansion": "Retrieval Augmented Generation",
            "confidence": "high",
        }
        assert a["expansion"] == "Retrieval Augmented Generation"

    def test_acronym_without_expansion(self):
        """Test Acronym without expansion (None)."""
        a: Acronym = {
            "text": "XYZ",
            "expansion": None,
            "confidence": "low",
        }
        assert a["expansion"] is None


class TestClarification:
    """Tests for Clarification TypedDict."""

    def test_clarification_not_needed(self):
        """Test Clarification when not needed."""
        c: Clarification = {
            "needed": False,
            "blocking": False,
            "reasons": [],
        }
        assert c["needed"] is False
        assert len(c["reasons"]) == 0

    def test_clarification_needed_blocking(self):
        """Test Clarification when needed and blocking."""
        c: Clarification = {
            "needed": True,
            "blocking": True,
            "reasons": ["missing_version", "ambiguous_acronym"],
        }
        assert c["needed"] is True
        assert c["blocking"] is True
        assert len(c["reasons"]) == 2

    def test_clarification_needed_non_blocking(self):
        """Test Clarification when needed but non-blocking."""
        c: Clarification = {
            "needed": True,
            "blocking": False,
            "reasons": ["missing_timeframe"],
        }
        assert c["needed"] is True
        assert c["blocking"] is False


class TestIntakeError:
    """Tests for IntakeError TypedDict."""

    def test_intake_error_minimal(self):
        """Test minimal IntakeError."""
        e: IntakeError = {
            "node": "normalize_gate",
            "type": "schema_validation",
            "message": "Missing field",
            "retryable": False,
            "details": None,
        }
        assert e["node"] == "normalize_gate"
        assert e["retryable"] is False

    def test_intake_error_with_details(self):
        """Test IntakeError with details."""
        e: IntakeError = {
            "node": "extract_signals",
            "type": "model_output_parse",
            "message": "Validation failed",
            "retryable": True,
            "details": {"validation_errors": ["field required"]},
        }
        assert e["details"] is not None
        assert "validation_errors" in e["details"]

    def test_intake_error_structure(self):
        """Test IntakeError has required fields."""
        e: IntakeError = {
            "node": "test",
            "type": "runtime_error",
            "message": "error",
            "retryable": True,
            "details": {},
        }
        # Verify all required fields present
        assert "node" in e
        assert "type" in e
        assert "message" in e
        assert "retryable" in e
        assert "details" in e
