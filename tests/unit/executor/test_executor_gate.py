# tests/unit/executor/test_executor_gate.py
"""Unit tests for executor_gate node."""

import pytest

from agentic_rag.executor.nodes.executor_gate import executor_gate


class TestExecutorGate:
    """Tests for executor_gate node."""

    def test_executor_gate_direct_answer_strategy(self, sample_plan):
        """Test executor_gate with direct_answer strategy (skips retrieval)."""
        plan = {**sample_plan, "strategy": "direct_answer"}
        state = {"plan": plan}

        result = executor_gate(state)

        assert result["continue_search"] is False
        assert result["final_evidence"] == []
        assert result["coverage"]["evidence_quality"] == "low"
        assert result["coverage"]["confidence"] == 0.0
        assert result["retrieval_report"]["skipped"] is True
        assert "direct_answer" in result["retrieval_report"]["reason"]

    def test_executor_gate_clarify_then_retrieve_strategy(self, sample_plan):
        """Test executor_gate with clarify_then_retrieve strategy (skips retrieval)."""
        plan = {**sample_plan, "strategy": "clarify_then_retrieve"}
        state = {"plan": plan}

        result = executor_gate(state)

        assert result["continue_search"] is False
        assert result["retrieval_report"]["skipped"] is True
        assert "clarify_then_retrieve" in result["retrieval_report"]["reason"]

    def test_executor_gate_defer_or_refuse_strategy(self, sample_plan):
        """Test executor_gate with defer_or_refuse strategy (skips retrieval)."""
        plan = {**sample_plan, "strategy": "defer_or_refuse"}
        state = {"plan": plan}

        result = executor_gate(state)

        assert result["continue_search"] is False
        assert result["retrieval_report"]["skipped"] is True
        assert "defer_or_refuse" in result["retrieval_report"]["reason"]

    def test_executor_gate_retrieve_then_answer_success(self, sample_plan):
        """Test executor_gate with retrieve_then_answer strategy (runs retrieval)."""
        state = {"plan": sample_plan}

        result = executor_gate(state)

        # Should initialize execution
        assert result["continue_search"] is True
        assert result["current_round_index"] == 0
        assert result["rounds"] == []
        assert result["evidence_pool"] == []
        assert result["retrieval_report"]["skipped"] is False

        # Should set execution context
        assert "execution_context" in result
        ctx = result["execution_context"]
        assert ctx["max_rounds"] == 1  # From sample_plan
        assert ctx["max_total_docs"] == 12  # From sample_plan
        assert ctx["confidence_threshold"] == 0.8  # From sample_plan
        assert ctx["no_new_information_rounds"] == 1

    def test_executor_gate_missing_retrieval_rounds(self):
        """Test error when retrieve_then_answer has no retrieval_rounds."""
        plan = {
            "strategy": "retrieve_then_answer",
            "retrieval_rounds": [],  # Empty
        }
        state = {"plan": plan}

        result = executor_gate(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "executor_gate"
        assert error["type"] == "schema_validation"
        assert "retrieval_rounds" in error["message"]
        assert result["continue_search"] is False

    def test_executor_gate_missing_plan(self):
        """Test executor_gate with missing plan."""
        state = {}

        result = executor_gate(state)

        # Missing plan should result in error (no strategy found)
        assert "errors" in result
        error = result["errors"][0]
        assert error["node"] == "executor_gate"

    def test_executor_gate_execution_context_defaults(self):
        """Test that execution_context uses defaults when stop_conditions missing."""
        plan = {
            "strategy": "retrieve_then_answer",
            "retrieval_rounds": [{"round_id": 0, "purpose": "recall"}],
            # No stop_conditions
        }
        state = {"plan": plan}

        result = executor_gate(state)

        ctx = result["execution_context"]
        assert ctx["max_rounds"] == 1  # Defaults to len(retrieval_rounds)
        assert ctx["max_total_docs"] == 12  # Default
        assert ctx["confidence_threshold"] is None  # Default
        assert ctx["no_new_information_rounds"] == 1  # Default

    def test_executor_gate_execution_context_custom_values(self):
        """Test execution_context with custom stop_conditions."""
        plan = {
            "strategy": "retrieve_then_answer",
            "retrieval_rounds": [
                {"round_id": 0, "purpose": "recall"},
                {"round_id": 1, "purpose": "precision"},
            ],
            "stop_conditions": {
                "max_rounds": 3,
                "max_total_docs": 20,
                "confidence_threshold": 0.9,
                "no_new_information_rounds": 2,
            },
        }
        state = {"plan": plan}

        result = executor_gate(state)

        ctx = result["execution_context"]
        assert ctx["max_rounds"] == 3
        assert ctx["max_total_docs"] == 20
        assert ctx["confidence_threshold"] == 0.9
        assert ctx["no_new_information_rounds"] == 2

    def test_executor_gate_max_rounds_minimum_one(self):
        """Test that max_rounds is at least 1 even if set to 0."""
        plan = {
            "strategy": "retrieve_then_answer",
            "retrieval_rounds": [{"round_id": 0, "purpose": "recall"}],
            "stop_conditions": {"max_rounds": 0},  # Invalid
        }
        state = {"plan": plan}

        result = executor_gate(state)

        ctx = result["execution_context"]
        assert ctx["max_rounds"] == 1  # Should be clamped to 1

    def test_executor_gate_coverage_structure_on_skip(self):
        """Test that coverage has all required fields when skipping retrieval."""
        plan = {"strategy": "direct_answer"}
        state = {"plan": plan}

        result = executor_gate(state)

        coverage = result["coverage"]
        assert "covered_entities" in coverage
        assert "missing_entities" in coverage
        assert "covered_subquestions" in coverage
        assert "missing_subquestions" in coverage
        assert "evidence_quality" in coverage
        assert "confidence" in coverage
        assert "contradictions" in coverage

        # All should be empty/low/0.0 for skipped retrieval
        assert coverage["covered_entities"] == []
        assert coverage["missing_entities"] == []
        assert coverage["evidence_quality"] == "low"
        assert coverage["confidence"] == 0.0
        assert coverage["contradictions"] == []

    def test_executor_gate_initializes_rounds_and_pool(self, sample_plan):
        """Test that executor_gate initializes empty rounds and evidence_pool."""
        state = {"plan": sample_plan}

        result = executor_gate(state)

        assert result["rounds"] == []
        assert result["evidence_pool"] == []
        assert isinstance(result["rounds"], list)
        assert isinstance(result["evidence_pool"], list)
