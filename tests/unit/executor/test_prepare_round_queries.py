# tests/unit/executor/test_prepare_round_queries.py
"""Unit tests for prepare_round_queries node."""

import pytest

from agentic_rag.executor.nodes.prepare_round_queries import (
    make_prepare_round_queries_node,
    _preserve_literal_terms,
)


class TestPreserveLiteralTerms:
    """Tests for _preserve_literal_terms helper function."""

    def test_preserve_no_constraints(self):
        """Test with no must_preserve_terms (returns queries unchanged)."""
        queries = ["query1", "query2", "query3"]
        result = _preserve_literal_terms(queries, [])

        assert result == queries

    def test_preserve_all_queries_match(self):
        """Test when all queries contain required terms."""
        queries = ["Azure OpenAI config", "Azure OpenAI setup"]
        must_preserve = ["Azure", "OpenAI"]

        result = _preserve_literal_terms(queries, must_preserve)

        # All queries should be kept (they all contain the terms)
        assert len(result) == 2
        assert "Azure OpenAI config" in result
        assert "Azure OpenAI setup" in result

    def test_preserve_some_queries_match(self):
        """Test when only some queries contain required terms."""
        queries = ["Azure OpenAI config", "cloud setup", "OpenAI docs"]
        must_preserve = ["Azure", "OpenAI"]

        result = _preserve_literal_terms(queries, must_preserve)

        # Only first query has both terms, should be first
        assert result[0] == "Azure OpenAI config"
        # Others should follow
        assert "cloud setup" in result
        assert "OpenAI docs" in result

    def test_preserve_no_queries_match(self):
        """Test when no queries contain all required terms."""
        queries = ["cloud setup", "configuration guide"]
        must_preserve = ["Azure", "OpenAI"]

        result = _preserve_literal_terms(queries, must_preserve)

        # Should create a fallback query with the terms
        assert len(result) >= 1
        first = result[0]
        assert "Azure" in first
        assert "OpenAI" in first

    def test_preserve_empty_queries(self):
        """Test with empty queries list."""
        queries = []
        must_preserve = ["Azure", "OpenAI"]

        result = _preserve_literal_terms(queries, must_preserve)

        # Should create a query with just the terms
        assert len(result) == 1
        assert "Azure" in result[0]
        assert "OpenAI" in result[0]

    def test_preserve_empty_terms_in_list(self):
        """Test handling of empty strings in must_preserve_terms."""
        queries = ["test query"]
        must_preserve = ["Azure", "", "OpenAI"]

        result = _preserve_literal_terms(queries, must_preserve)

        # Should handle empty strings gracefully
        assert len(result) >= 1


class TestPrepareRoundQueries:
    """Tests for prepare_round_queries node."""

    def test_prepare_basic_query_variants(self, mock_hyde, sample_plan, sample_executor_state):
        """Test basic query preparation from round spec."""
        node = make_prepare_round_queries_node(mock_hyde)

        result = node(sample_executor_state)

        assert "round_queries" in result
        queries = result["round_queries"]
        assert len(queries) > 0
        # Should use query_variants from round spec
        assert queries[0] == "Azure OpenAI configuration"

    def test_prepare_fallback_to_normalized_query(self, mock_hyde, sample_plan):
        """Test fallback to normalized_query when no query_variants."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["query_variants"] = []

        state = {
            "plan": plan,
            "normalized_query": "fallback query",
            "current_round_index": 0,
        }

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        queries = result["round_queries"]
        assert "fallback query" in queries

    def test_prepare_out_of_bounds_round_index(self, mock_hyde, sample_executor_state):
        """Test when current_round_index exceeds number of rounds."""
        state = {**sample_executor_state, "current_round_index": 99}

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Should stop search
        assert result["continue_search"] is False

    def test_prepare_with_hyde_enabled(self, mock_hyde, sample_plan):
        """Test query preparation with HyDE enabled."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["use_hyde"] = True
        plan["literal_constraints"] = {}  # No constraints

        state = {
            "plan": plan,
            "normalized_query": "test query",
            "current_round_index": 0,
        }

        # Mock HyDE responses
        mock_hyde.synthesize.return_value = "synthetic answer"
        mock_hyde.derive_queries.return_value = ["derived1", "derived2"]

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Should call HyDE
        assert mock_hyde.synthesize.called
        assert mock_hyde.derive_queries.called

        queries = result["round_queries"]
        # Should include original query + derived queries
        assert "test query" in queries
        assert "derived1" in queries
        assert "derived2" in queries

    def test_prepare_hyde_disabled_by_literal_constraints(self, mock_hyde, sample_plan):
        """Test that HyDE is disabled when literal constraints present."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["use_hyde"] = True
        plan["literal_constraints"] = {
            "must_preserve_terms": ["Azure", "OpenAI"],
            "must_match_exactly": False,
        }

        state = {
            "plan": plan,
            "normalized_query": "Azure OpenAI setup",
            "current_round_index": 0,
        }

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Should NOT call HyDE due to must_preserve_terms
        assert not mock_hyde.synthesize.called
        assert not mock_hyde.derive_queries.called

    def test_prepare_hyde_disabled_by_must_match_exactly(self, mock_hyde, sample_plan):
        """Test that HyDE is disabled when must_match_exactly is True."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["use_hyde"] = True
        plan["literal_constraints"] = {
            "must_match_exactly": True,
        }

        state = {
            "plan": plan,
            "normalized_query": "test query",
            "current_round_index": 0,
        }

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Should NOT call HyDE
        assert not mock_hyde.synthesize.called
        assert not mock_hyde.derive_queries.called

    def test_prepare_with_must_preserve_terms(self, mock_hyde, sample_plan):
        """Test query preparation with must_preserve_terms."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["query_variants"] = [
            "Azure OpenAI config",
            "cloud setup",
        ]
        plan["literal_constraints"] = {
            "must_preserve_terms": ["Azure", "OpenAI"],
        }

        state = {
            "plan": plan,
            "normalized_query": "Azure OpenAI configuration",
            "current_round_index": 0,
        }

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        queries = result["round_queries"]
        # First query should contain both terms
        assert "Azure" in queries[0]
        assert "OpenAI" in queries[0]

    def test_prepare_missing_plan(self, mock_hyde):
        """Test with missing plan."""
        state = {"current_round_index": 0}

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Should stop search due to no rounds
        assert result["continue_search"] is False

    def test_prepare_round_index_zero(self, mock_hyde, sample_executor_state):
        """Test with round index 0 (first round)."""
        state = {**sample_executor_state, "current_round_index": 0}

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        assert "round_queries" in result
        assert len(result["round_queries"]) > 0

    def test_prepare_hyde_with_context(self, mock_hyde, sample_plan):
        """Test that HyDE receives proper context."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["use_hyde"] = True
        plan["literal_constraints"] = {}

        state = {
            "plan": plan,
            "normalized_query": "test query",
            "current_round_index": 0,
        }

        mock_hyde.synthesize.return_value = "synthetic"
        mock_hyde.derive_queries.return_value = ["derived"]

        node = make_prepare_round_queries_node(mock_hyde)
        result = node(state)

        # Verify synthesize was called with correct args
        call_args = mock_hyde.synthesize.call_args
        assert call_args[1]["query"] == "test query"
        assert "plan" in call_args[1]["context"]

        # Verify derive_queries was called
        call_args = mock_hyde.derive_queries.call_args
        assert call_args[1]["original_query"] == "test query"
        assert call_args[1]["synthetic_answer"] == "synthetic"
        assert call_args[1]["max_queries"] == 4
