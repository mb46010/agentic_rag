# tests/unit/executor/test_merge_candidates.py
"""Unit tests for merge_candidates node."""

import pytest
from dataclasses import replace

from agentic_rag.executor.nodes.merge_candidates import make_merge_candidates_node, _dedupe
from agentic_rag.executor.state import Candidate, CandidateKey


class TestDedupe:
    """Tests for _dedupe helper function."""

    def test_dedupe_no_duplicates(self, sample_candidates):
        """Test deduplication with no duplicates."""
        result = _dedupe(sample_candidates)

        # All unique candidates, should have same count
        assert len(result) == len(sample_candidates)

    def test_dedupe_with_duplicates(self, sample_candidate):
        """Test deduplication with duplicate keys."""
        # Create duplicates with same key (clear other scores to test bm25_score comparison)
        dup1 = replace(sample_candidate, bm25_score=0.5, vector_score=None, rrf_score=None, rerank_score=None)
        dup2 = replace(sample_candidate, bm25_score=0.9, vector_score=None, rrf_score=None, rerank_score=None)  # Higher score
        dup3 = replace(sample_candidate, bm25_score=0.7, vector_score=None, rrf_score=None, rerank_score=None)

        candidates = [dup1, dup2, dup3]
        result = _dedupe(candidates)

        # Should have only 1 candidate (best score)
        assert len(result) == 1
        assert result[0].bm25_score == 0.9

    def test_dedupe_prefers_rerank_score(self, sample_candidate):
        """Test that deduplication prefers rerank_score over other scores."""
        c1 = replace(sample_candidate, bm25_score=0.9, rerank_score=None)
        c2 = replace(sample_candidate, bm25_score=0.5, rerank_score=0.95)

        result = _dedupe([c1, c2])

        # Should keep c2 (has rerank_score)
        assert len(result) == 1
        assert result[0].rerank_score == 0.95

    def test_dedupe_score_precedence(self, sample_candidate):
        """Test score precedence: rerank > rrf > vector > bm25."""
        # Same key, different scores
        c1 = replace(sample_candidate, bm25_score=0.9, vector_score=None, rrf_score=None, rerank_score=None)
        c2 = replace(sample_candidate, bm25_score=None, vector_score=0.85, rrf_score=None, rerank_score=None)
        c3 = replace(sample_candidate, bm25_score=None, vector_score=None, rrf_score=0.88, rerank_score=None)
        c4 = replace(sample_candidate, bm25_score=None, vector_score=None, rrf_score=None, rerank_score=0.95)

        # Test with different orders
        result = _dedupe([c1, c2, c3, c4])
        assert result[0].rerank_score == 0.95

        result = _dedupe([c4, c3, c2, c1])
        assert result[0].rerank_score == 0.95

    def test_dedupe_empty_list(self):
        """Test deduplication with empty list."""
        result = _dedupe([])
        assert result == []


class TestMergeCandidates:
    """Tests for merge_candidates node."""

    def test_merge_basic(self, mock_fusion, sample_plan, sample_candidates):
        """Test basic candidate merging."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_raw": sample_candidates,
        }

        # Mock fusion to return candidates
        mock_fusion.rrf.return_value = sample_candidates

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        assert "round_candidates_merged" in result
        merged = result["round_candidates_merged"]
        assert len(merged) > 0

    def test_merge_empty_raw_candidates(self, mock_fusion, sample_plan):
        """Test merging with no raw candidates."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_raw": [],
        }

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        merged = result["round_candidates_merged"]
        assert merged == []

        # RRF should not be called
        assert not mock_fusion.rrf.called

    def test_merge_with_rrf_enabled(self, mock_fusion, sample_plan, sample_candidates):
        """Test merging with RRF enabled."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rrf"] = True

        # Add query and mode to candidates for grouping
        enriched = [
            replace(sample_candidates[0], query="q1", mode="bm25"),
            replace(sample_candidates[1], query="q1", mode="vector"),
            replace(sample_candidates[2], query="q2", mode="bm25"),
        ]

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_raw": enriched,
        }

        mock_fusion.rrf.return_value = enriched

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        # RRF should be called
        assert mock_fusion.rrf.called
        call_args = mock_fusion.rrf.call_args
        assert "ranked_lists" in call_args[1]
        assert "k" in call_args[1]
        assert "rrf_k" in call_args[1]

    def test_merge_with_rrf_disabled(self, mock_fusion, sample_plan, sample_candidates):
        """Test merging with RRF disabled."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["rrf"] = False

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_raw": sample_candidates,
        }

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        # RRF should not be called
        assert not mock_fusion.rrf.called

        # Should still return merged (deduplicated) candidates
        merged = result["round_candidates_merged"]
        assert len(merged) > 0

    def test_merge_groups_by_query_and_mode(self, mock_fusion, sample_plan, sample_candidate):
        """Test that candidates are grouped by (query, mode) for RRF."""
        # Create candidates with different query/mode combinations
        c1 = replace(sample_candidate, key=CandidateKey("d1", "c1"), query="q1", mode="bm25")
        c2 = replace(sample_candidate, key=CandidateKey("d2", "c2"), query="q1", mode="vector")
        c3 = replace(sample_candidate, key=CandidateKey("d3", "c3"), query="q2", mode="bm25")

        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_raw": [c1, c2, c3],
        }

        mock_fusion.rrf.return_value = [c1, c2, c3]

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        # RRF should be called with multiple ranked lists
        assert mock_fusion.rrf.called
        call_args = mock_fusion.rrf.call_args
        ranked_lists = call_args[1]["ranked_lists"]
        # Should have 3 lists: (q1, bm25), (q1, vector), (q2, bm25)
        assert len(ranked_lists) == 3

    def test_merge_deduplicates_after_rrf(self, mock_fusion, sample_plan, sample_candidate):
        """Test that candidates are deduplicated after RRF fusion."""
        # RRF might return duplicates
        dup1 = replace(sample_candidate, query="q1", mode="bm25", rrf_score=0.5)
        dup2 = replace(sample_candidate, query="q1", mode="vector", rrf_score=0.9)

        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_raw": [dup1, dup2],
        }

        mock_fusion.rrf.return_value = [dup1, dup2]

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        merged = result["round_candidates_merged"]
        # Should be deduplicated to 1 candidate (best rrf_score)
        assert len(merged) == 1
        assert merged[0].rrf_score == 0.9

    def test_merge_sorts_by_rrf_score(self, mock_fusion, sample_plan):
        """Test that merged candidates are sorted by rrf_score."""
        c1 = Candidate(
            key=CandidateKey("d1", "c1"),
            text="text1",
            query="q1",
            mode="bm25",
            rrf_score=0.5,
        )
        c2 = Candidate(
            key=CandidateKey("d2", "c2"),
            text="text2",
            query="q1",
            mode="vector",
            rrf_score=0.9,
        )
        c3 = Candidate(
            key=CandidateKey("d3", "c3"),
            text="text3",
            query="q2",
            mode="bm25",
            rrf_score=0.7,
        )

        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_raw": [c1, c2, c3],
        }

        mock_fusion.rrf.return_value = [c1, c2, c3]

        node = make_merge_candidates_node(mock_fusion)
        result = node(state)

        merged = result["round_candidates_merged"]
        # Should be sorted by rrf_score descending
        assert merged[0].rrf_score == 0.9
        assert merged[1].rrf_score == 0.7
        assert merged[2].rrf_score == 0.5
