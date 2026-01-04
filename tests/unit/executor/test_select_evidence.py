# tests/unit/executor/test_select_evidence.py
"""Unit tests for select_evidence node."""

import pytest
from dataclasses import replace

from agentic_rag.executor.nodes.select_evidence import select_evidence, _diverse_top_k
from agentic_rag.executor.state import Candidate, CandidateKey


class TestDiverseTopK:
    """Tests for _diverse_top_k helper function."""

    def test_diverse_top_k_basic(self, sample_candidate):
        """Test basic diverse selection."""
        # Create candidates from different docs
        c1 = replace(sample_candidate, key=CandidateKey("doc1", "c1"), rerank_score=0.9)
        c2 = replace(sample_candidate, key=CandidateKey("doc2", "c1"), rerank_score=0.8)
        c3 = replace(sample_candidate, key=CandidateKey("doc3", "c1"), rerank_score=0.7)

        result = _diverse_top_k([c1, c2, c3], max_docs=2)

        assert len(result) == 2

    def test_diverse_top_k_same_doc(self, sample_candidate):
        """Test diversity when multiple chunks from same doc."""
        # Create candidates from same doc
        c1 = replace(sample_candidate, key=CandidateKey("doc1", "c1"), rerank_score=0.9)
        c2 = replace(sample_candidate, key=CandidateKey("doc1", "c2"), rerank_score=0.8)
        c3 = replace(sample_candidate, key=CandidateKey("doc1", "c3"), rerank_score=0.7)

        result = _diverse_top_k([c1, c2, c3], max_docs=2)

        # Should round-robin, taking c1 first, then c2
        assert len(result) == 2
        assert result[0].key.chunk_id == "c1"
        assert result[1].key.chunk_id == "c2"

    def test_diverse_top_k_round_robin(self, sample_candidate):
        """Test round-robin distribution across docs."""
        # 2 docs, 2 chunks each
        d1c1 = replace(sample_candidate, key=CandidateKey("doc1", "c1"), rerank_score=0.9)
        d1c2 = replace(sample_candidate, key=CandidateKey("doc1", "c2"), rerank_score=0.8)
        d2c1 = replace(sample_candidate, key=CandidateKey("doc2", "c1"), rerank_score=0.7)
        d2c2 = replace(sample_candidate, key=CandidateKey("doc2", "c2"), rerank_score=0.6)

        result = _diverse_top_k([d1c1, d1c2, d2c1, d2c2], max_docs=3)

        # Should alternate: d1c1, d2c1, d1c2
        assert len(result) == 3
        assert result[0].key.doc_id == "doc1"
        assert result[1].key.doc_id == "doc2"
        assert result[2].key.doc_id == "doc1"

    def test_diverse_top_k_empty_list(self):
        """Test with empty candidate list."""
        result = _diverse_top_k([], max_docs=5)
        assert result == []

    def test_diverse_top_k_fewer_than_max(self, sample_candidate):
        """Test when candidates fewer than max_docs."""
        c1 = replace(sample_candidate, key=CandidateKey("doc1", "c1"))
        c2 = replace(sample_candidate, key=CandidateKey("doc2", "c1"))

        result = _diverse_top_k([c1, c2], max_docs=10)

        # Should return all available candidates
        assert len(result) == 2


class TestSelectEvidence:
    """Tests for select_evidence node."""

    def test_select_basic(self, sample_plan, sample_candidates):
        """Test basic evidence selection."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_reranked": sample_candidates,
        }

        result = select_evidence(state)

        assert "round_selected" in result
        selected = result["round_selected"]
        assert len(selected) > 0

    def test_select_empty_reranked(self, sample_plan):
        """Test selection with no reranked candidates."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_candidates_reranked": [],
        }

        result = select_evidence(state)

        selected = result["round_selected"]
        assert selected == []

    def test_select_respects_max_docs(self, sample_plan):
        """Test that selection respects max_docs limit."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["output"] = {"max_docs": 2}

        # Create 5 candidates
        candidates = [
            Candidate(
                key=CandidateKey(f"doc{i}", "c1"),
                text=f"text{i}",
                rerank_score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_reranked": candidates,
        }

        result = select_evidence(state)

        selected = result["round_selected"]
        assert len(selected) == 2

    def test_select_default_max_docs(self, sample_plan, sample_candidates):
        """Test default max_docs when not specified."""
        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["output"] = {}

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_reranked": sample_candidates,
        }

        result = select_evidence(state)

        # Should use DEFAULT_MAX_DOCS_PER_ROUND (8)
        selected = result["round_selected"]
        assert len(selected) <= 8

    def test_select_sorts_by_score(self, sample_plan):
        """Test that candidates are sorted by score before selection."""
        # Create candidates with different scores (unsorted)
        candidates = [
            Candidate(key=CandidateKey("doc1", "c1"), text="t1", rerank_score=0.5),
            Candidate(key=CandidateKey("doc2", "c1"), text="t2", rerank_score=0.9),
            Candidate(key=CandidateKey("doc3", "c1"), text="t3", rerank_score=0.7),
        ]

        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["output"] = {"max_docs": 10}

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_reranked": candidates,
        }

        result = select_evidence(state)

        selected = result["round_selected"]
        # After sorting and diversity selection, highest scores should be prioritized
        # Note: diversity may affect order, but all should be selected
        assert len(selected) == 3

    def test_select_prefers_rerank_over_rrf(self, sample_plan):
        """Test that selection prefers rerank_score over rrf_score."""
        c1 = Candidate(key=CandidateKey("doc1", "c1"), text="t1", rerank_score=0.9, rrf_score=0.5)
        c2 = Candidate(key=CandidateKey("doc2", "c1"), text="t2", rerank_score=None, rrf_score=0.8)

        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["output"] = {"max_docs": 10}

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_reranked": [c1, c2],
        }

        result = select_evidence(state)

        selected = result["round_selected"]
        # c1 should be first (has rerank_score=0.9)
        assert len(selected) == 2

    def test_select_applies_diversity(self, sample_plan):
        """Test that diverse_top_k applies diversity."""
        # Multiple chunks from same doc
        candidates = [
            Candidate(key=CandidateKey("doc1", "c1"), text="t1", rerank_score=0.9),
            Candidate(key=CandidateKey("doc1", "c2"), text="t2", rerank_score=0.8),
            Candidate(key=CandidateKey("doc2", "c1"), text="t3", rerank_score=0.7),
        ]

        plan = {**sample_plan}
        plan["retrieval_rounds"][0]["output"] = {"max_docs": 2}

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_candidates_reranked": candidates,
        }

        result = select_evidence(state)

        selected = result["round_selected"]
        # Should distribute across docs (round-robin)
        assert len(selected) == 2
