# tests/unit/executor/test_grade_coverage.py
"""Unit tests for grade_coverage node."""

import pytest

from agentic_rag.executor.nodes.grade_coverage import make_grade_coverage_node


class TestGradeCoverage:
    """Tests for grade_coverage node."""

    def test_grade_basic(self, mock_grader, sample_plan, sample_executor_state, sample_candidates):
        """Test basic coverage grading."""
        state = {
            **sample_executor_state,
            "round_selected": sample_candidates,
        }

        # Mock grader response
        mock_grader.grade.return_value = {
            "covered_entities": ["Azure OpenAI"],
            "missing_entities": [],
            "covered_subquestions": [],
            "missing_subquestions": [],
            "evidence_quality": "high",
            "confidence": 0.9,
            "contradictions": [],
        }

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        assert "coverage" in result
        coverage = result["coverage"]
        assert coverage["confidence"] == 0.9
        assert coverage["evidence_quality"] == "high"

        # Verify grader was called
        assert mock_grader.grade.called

    def test_grade_calls_grader_with_correct_args(self, mock_grader, sample_plan, sample_candidates):
        """Test that grader is called with correct arguments."""
        state = {
            "plan": sample_plan,
            "normalized_query": "test query",
            "round_selected": sample_candidates,
            "constraints": {"domain": ["azure"]},
            "guardrails": {"sensitivity": "normal"},
        }

        mock_grader.grade.return_value = {
            "covered_entities": [],
            "missing_entities": [],
            "evidence_quality": "low",
            "confidence": 0.0,
        }

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        # Verify grader was called with expected args
        call_args = mock_grader.grade.call_args
        assert call_args[1]["plan"] == sample_plan
        assert call_args[1]["normalized_query"] == "test query"
        assert call_args[1]["selected_evidence"] == sample_candidates
        assert "context" in call_args[1]
        assert call_args[1]["context"]["constraints"] == {"domain": ["azure"]}
        assert call_args[1]["context"]["guardrails"] == {"sensitivity": "normal"}

    def test_grade_empty_selected(self, mock_grader, sample_plan):
        """Test grading with no selected evidence."""
        state = {
            "plan": sample_plan,
            "normalized_query": "test query",
            "round_selected": [],
        }

        mock_grader.grade.return_value = {
            "covered_entities": [],
            "missing_entities": [],
            "evidence_quality": "low",
            "confidence": 0.0,
            "contradictions": [],
        }

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        # Should still call grader (with empty evidence)
        assert mock_grader.grade.called
        assert result["coverage"]["confidence"] == 0.0

    def test_grade_missing_constraints_and_guardrails(self, mock_grader, sample_plan, sample_candidates):
        """Test grading when constraints and guardrails missing."""
        state = {
            "plan": sample_plan,
            "normalized_query": "test",
            "round_selected": sample_candidates,
            # Missing constraints and guardrails
        }

        mock_grader.grade.return_value = {
            "covered_entities": [],
            "missing_entities": [],
            "evidence_quality": "medium",
            "confidence": 0.5,
        }

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        # Should use empty dicts for missing context
        call_args = mock_grader.grade.call_args
        assert call_args[1]["context"]["constraints"] == {}
        assert call_args[1]["context"]["guardrails"] == {}

    def test_grade_missing_normalized_query(self, mock_grader, sample_plan, sample_candidates):
        """Test grading when normalized_query missing."""
        state = {
            "plan": sample_plan,
            # Missing normalized_query
            "round_selected": sample_candidates,
        }

        mock_grader.grade.return_value = {
            "covered_entities": [],
            "missing_entities": [],
            "evidence_quality": "low",
            "confidence": 0.0,
        }

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        # Should use empty string
        call_args = mock_grader.grade.call_args
        assert call_args[1]["normalized_query"] == ""

    def test_grade_returns_coverage_dict(self, mock_grader, sample_plan, sample_candidates):
        """Test that grade returns coverage dict."""
        state = {
            "plan": sample_plan,
            "normalized_query": "test",
            "round_selected": sample_candidates,
        }

        expected_coverage = {
            "covered_entities": ["entity1", "entity2"],
            "missing_entities": ["entity3"],
            "covered_subquestions": ["q1"],
            "missing_subquestions": ["q2"],
            "evidence_quality": "medium",
            "confidence": 0.7,
            "contradictions": ["contradiction1"],
        }

        mock_grader.grade.return_value = expected_coverage

        node = make_grade_coverage_node(mock_grader)
        result = node(state)

        assert result["coverage"] == expected_coverage
