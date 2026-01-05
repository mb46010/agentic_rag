# tests/unit/answer/test_answer_gate.py
"""Unit tests for answer_gate node."""

import pytest

from agentic_rag.answer.nodes.answer_gate import make_answer_gate_node


class TestAnswerGate:
    """Tests for answer_gate node."""

    def test_refuse_when_strategy_defer_or_refuse(self, sample_answer_state):
        """Should refuse when strategy is defer_or_refuse."""
        state = {**sample_answer_state, "plan": {"strategy": "defer_or_refuse"}}

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "refuse"
        assert result["answer_meta"]["mode"] == "refuse"
        assert result["answer_meta"]["refusal"] is True
        assert result["answer_meta"]["asked_clarification"] is False

    def test_refuse_when_sensitivity_restricted(self, sample_answer_state):
        """Should refuse when sensitivity is restricted."""
        state = {
            **sample_answer_state,
            "guardrails": {"sensitivity": "restricted"},
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "refuse"
        assert result["answer_meta"]["refusal"] is True

    def test_clarify_when_strategy_clarify_then_retrieve(self, sample_answer_state):
        """Should clarify when strategy is clarify_then_retrieve."""
        state = {
            **sample_answer_state,
            "plan": {"strategy": "clarify_then_retrieve"},
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "clarify"
        assert result["answer_meta"]["mode"] == "clarify"
        assert result["answer_meta"]["asked_clarification"] is True
        assert result["answer_meta"]["refusal"] is False

    def test_clarify_when_coverage_has_contradictions(self, sample_answer_state):
        """Should clarify when coverage has contradictions."""
        state = {
            **sample_answer_state,
            "coverage": {
                "confidence": 0.9,
                "covered": ["item1"],
                "missing": [],
                "contradictions": ["contradiction1"],
            },
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "clarify"
        assert result["answer_meta"]["asked_clarification"] is True

    def test_clarify_when_coverage_has_missing_items(self, sample_answer_state):
        """Should clarify when coverage has missing items."""
        state = {
            **sample_answer_state,
            "coverage": {
                "confidence": 0.9,
                "covered": ["item1"],
                "missing": ["missing_item"],
                "contradictions": [],
            },
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "clarify"

    def test_clarify_when_confidence_below_threshold(self, sample_answer_state):
        """Should clarify when confidence is below threshold."""
        state = {
            **sample_answer_state,
            "plan": {
                "strategy": "retrieve_then_answer",
                "stop_conditions": {"confidence_threshold": 0.8},
            },
            "coverage": {
                "confidence": 0.5,  # Below threshold
                "covered": ["item1"],
                "missing": [],
                "contradictions": [],
            },
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "clarify"

    def test_answer_when_coverage_is_good(self, sample_answer_state):
        """Should answer when coverage is good."""
        state = {
            **sample_answer_state,
            "plan": {
                "strategy": "retrieve_then_answer",
                "stop_conditions": {"confidence_threshold": 0.7},
            },
            "coverage": {
                "confidence": 0.85,
                "covered": ["LangGraph"],
                "missing": [],
                "contradictions": [],
            },
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "answer"
        assert result["answer_meta"]["mode"] == "answer"
        assert result["answer_meta"]["refusal"] is False
        assert result["answer_meta"]["asked_clarification"] is False
        assert result["answer_meta"]["coverage_confidence"] == 0.85

    def test_answer_when_no_confidence_threshold(self, sample_answer_state):
        """Should answer when no confidence threshold is set."""
        state = {
            **sample_answer_state,
            "plan": {"strategy": "retrieve_then_answer", "stop_conditions": {}},
            "coverage": {
                "confidence": 0.5,
                "covered": ["item1"],
                "missing": [],
                "contradictions": [],
            },
        }

        node = make_answer_gate_node()
        result = node(state)

        assert result["answer_mode"] == "answer"

    def test_handles_missing_coverage_gracefully(self, sample_answer_state):
        """Should handle missing coverage dict gracefully."""
        state = {
            **sample_answer_state,
            "coverage": None,
            "plan": {"strategy": "retrieve_then_answer", "stop_conditions": {}},  # No threshold
        }

        node = make_answer_gate_node()
        result = node(state)

        # Should default to answer mode with 0 confidence
        assert result["answer_mode"] == "answer"
        assert result["answer_meta"]["coverage_confidence"] == 0.0

    def test_handles_invalid_coverage_gracefully(self, sample_answer_state):
        """Should handle invalid coverage data gracefully."""
        state = {**sample_answer_state, "coverage": "invalid"}

        node = make_answer_gate_node()
        result = node(state)

        # Should not crash, should use default coverage
        assert "answer_mode" in result

    def test_sensitivity_from_plan_safety(self, sample_answer_state):
        """Should use sensitivity from plan.safety if present."""
        state = {
            **sample_answer_state,
            "plan": {
                "strategy": "retrieve_then_answer",
                "safety": {"sensitivity": "restricted"},
            },
            "guardrails": {"sensitivity": "normal"},  # This should be overridden
        }

        node = make_answer_gate_node()
        result = node(state)

        # Plan safety should take precedence
        assert result["answer_mode"] == "refuse"

    def test_handles_missing_plan(self, sample_answer_state):
        """Should handle missing plan gracefully."""
        state = {**sample_answer_state, "plan": None}

        node = make_answer_gate_node()
        result = node(state)

        # Should not crash
        assert "answer_mode" in result

    def test_handles_missing_guardrails(self, sample_answer_state):
        """Should handle missing guardrails gracefully."""
        state = {**sample_answer_state, "guardrails": None}

        node = make_answer_gate_node()
        result = node(state)

        # Should not crash
        assert "answer_mode" in result
