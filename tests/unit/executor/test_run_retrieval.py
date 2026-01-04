# tests/unit/executor/test_run_retrieval.py
"""Unit tests for run_retrieval node."""

import pytest

from agentic_rag.executor.nodes.run_retrieval import make_run_retrieval_node
from agentic_rag.executor.state import Candidate, CandidateKey


class TestRunRetrieval:
    """Tests for run_retrieval node."""

    def test_run_retrieval_basic(self, mock_retriever, sample_plan, sample_candidate):
        """Test basic retrieval with single query and mode."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        # Mock retriever to return candidates
        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        assert "round_candidates_raw" in result
        candidates = result["round_candidates_raw"]
        assert len(candidates) > 0

        # Verify retriever was called
        assert mock_retriever.search.called
        call_args = mock_retriever.search.call_args
        assert call_args[1]["query"] == "test query"

    def test_run_retrieval_enriches_provenance(self, mock_retriever, sample_plan, sample_candidate):
        """Test that retrieval enriches candidates with provenance fields."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        candidates = result["round_candidates_raw"]
        enriched = candidates[0]

        # Should have provenance fields
        assert enriched.round_id == 0
        assert enriched.query == "test query"
        assert enriched.mode is not None

    def test_run_retrieval_missing_queries(self, mock_retriever, sample_plan):
        """Test error handling when round_queries missing."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            # Missing round_queries
        }

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["node"] == "run_retrieval"
        assert error["type"] == "schema_validation"
        assert "round_queries" in error["message"]

    def test_run_retrieval_empty_queries(self, mock_retriever, sample_plan):
        """Test error handling when round_queries empty."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": [],
        }

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        assert "errors" in result

    def test_run_retrieval_multiple_queries(self, mock_retriever, sample_plan, sample_candidate):
        """Test retrieval with multiple queries."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["query1", "query2", "query3"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        # Should be called 3 times (one per query)
        assert mock_retriever.search.call_count == 3

        candidates = result["round_candidates_raw"]
        # 3 queries * 1 result each = 3 candidates
        assert len(candidates) == 3

    def test_run_retrieval_multiple_modes(self, mock_retriever, sample_plan, sample_candidate):
        """Test retrieval with multiple retrieval modes."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["retrieval_modes"] = [
            {"type": "bm25", "k": 10},
            {"type": "vector", "k": 10},
            {"type": "hybrid", "k": 10, "alpha": 0.5},
        ]

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        # Should be called 3 times (one per mode)
        assert mock_retriever.search.call_count == 3

        candidates = result["round_candidates_raw"]
        # 1 query * 3 modes * 1 result = 3 candidates
        assert len(candidates) == 3

    def test_run_retrieval_with_filters(self, mock_retriever, sample_plan, sample_candidate):
        """Test that filters are passed to retriever."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["filters"] = {
            "domains": ["azure"],
            "doc_types": ["documentation"],
        }

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        # Verify filters were passed
        call_args = mock_retriever.search.call_args
        filters = call_args[1]["filters"]
        assert filters["domains"] == ["azure"]
        assert filters["doc_types"] == ["documentation"]

    def test_run_retrieval_default_mode_and_k(self, mock_retriever, sample_plan, sample_candidate):
        """Test default retrieval mode and k when not specified."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["retrieval_modes"] = []

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        # Should use default mode (hybrid)
        call_args = mock_retriever.search.call_args
        assert call_args[1]["mode"] == "hybrid"
        assert call_args[1]["k"] == 20  # DEFAULT_RETRIEVAL_K

    def test_run_retrieval_custom_k_and_alpha(self, mock_retriever, sample_plan, sample_candidate):
        """Test custom k and alpha parameters."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["retrieval_modes"] = [
            {"type": "hybrid", "k": 50, "alpha": 0.7}
        ]

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = [sample_candidate]

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        call_args = mock_retriever.search.call_args
        assert call_args[1]["k"] == 50
        assert call_args[1]["alpha"] == 0.7

    def test_run_retrieval_no_results(self, mock_retriever, sample_plan):
        """Test when retriever returns no results."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["test query"],
        }

        mock_retriever.search.return_value = []

        node = make_run_retrieval_node(mock_retriever)
        result = node(state)

        candidates = result["round_candidates_raw"]
        assert candidates == []
