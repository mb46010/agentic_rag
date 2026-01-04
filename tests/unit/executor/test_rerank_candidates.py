# tests/unit/executor/test_rerank_candidates.py
"""Unit tests for rerank_candidates node."""

import pytest

from agentic_rag.executor.nodes.rerank_candidates import make_rerank_candidates_node


class TestRerankCandidates:
    """Tests for rerank_candidates node."""

    def test_rerank_basic(self, mock_reranker, sample_plan, sample_executor_state, sample_candidates):
        """Test basic candidate reranking."""
        state = {
            **sample_executor_state,
            "round_candidates_merged": sample_candidates,
        }

        # Mock reranker to return reranked candidates
        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        assert "round_candidates_reranked" in result
        reranked = result["round_candidates_reranked"]
        assert len(reranked) > 0

        # Verify reranker was called
        assert mock_reranker.rerank.called

    def test_rerank_empty_merged_candidates(self, mock_reranker, sample_plan):
        """Test reranking with no merged candidates."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_merged": [],
        }

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        reranked = result["round_candidates_reranked"]
        assert reranked == []

        # Reranker should not be called
        assert not mock_reranker.rerank.called

    def test_rerank_enabled(self, mock_reranker, sample_plan, sample_candidates):
        """Test reranking when enabled in rerank spec."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rerank"] = {
            "enabled": True,
            "model": "cross_encoder",
            "rerank_top_k": 60,
        }

        state = {
            "plan": plan,
            "normalized_query": "test query",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        # Reranker should be called
        assert mock_reranker.rerank.called
        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["candidates"] == sample_candidates
        assert call_args[1]["top_k"] == 60

    def test_rerank_disabled(self, mock_reranker, sample_plan, sample_candidates):
        """Test that reranking is skipped when disabled."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rerank"] = {"enabled": False}

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        # Reranker should not be called
        assert not mock_reranker.rerank.called

        # Should return merged candidates unchanged
        reranked = result["round_candidates_reranked"]
        assert reranked == sample_candidates

    def test_rerank_default_enabled(self, mock_reranker, sample_plan, sample_candidates):
        """Test that reranking is enabled by default."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rerank"] = {}  # No explicit enabled field

        state = {
            "plan": plan,
            "normalized_query": "test",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        # Should be enabled by default
        assert mock_reranker.rerank.called

    def test_rerank_default_top_k(self, mock_reranker, sample_plan, sample_candidates):
        """Test default top_k when not specified."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rerank"] = {"enabled": True}

        state = {
            "plan": plan,
            "normalized_query": "test",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["top_k"] == 60  # DEFAULT_RERANK_TOP_K

    def test_rerank_custom_top_k(self, mock_reranker, sample_plan, sample_candidates):
        """Test custom top_k value."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rerank"] = {
            "enabled": True,
            "rerank_top_k": 100,
        }

        state = {
            "plan": plan,
            "normalized_query": "test",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["top_k"] == 100

    def test_rerank_passes_context(self, mock_reranker, sample_plan, sample_candidates):
        """Test that reranker receives context with plan."""
        state = {
            "plan": sample_plan,
            "normalized_query": "test",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        call_args = mock_reranker.rerank.call_args
        context = call_args[1]["context"]
        assert "plan" in context
        assert context["plan"] == sample_plan

    def test_rerank_uses_normalized_query(self, mock_reranker, sample_plan, sample_candidates):
        """Test that reranker uses normalized_query from state."""
        state = {
            "plan": sample_plan,
            "normalized_query": "my normalized query",
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["query"] == "my normalized query"

    def test_rerank_missing_normalized_query(self, mock_reranker, sample_plan, sample_candidates):
        """Test reranking when normalized_query missing (uses empty string)."""
        state = {
            "plan": sample_plan,
            # Missing normalized_query
            "current_round_index": 0,
            "round_candidates_merged": sample_candidates,
        }

        mock_reranker.rerank.return_value = sample_candidates

        node = make_rerank_candidates_node(mock_reranker)
        result = node(state)

        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["query"] == ""
