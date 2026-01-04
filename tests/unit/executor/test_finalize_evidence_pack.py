# tests/unit/executor/test_finalize_evidence_pack.py
"""Unit tests for finalize_evidence_pack node."""

import pytest

from agentic_rag.executor.nodes.finalize_evidence_pack import finalize_evidence_pack
from agentic_rag.executor.state import Candidate, CandidateKey, RoundResult


class TestFinalizeEvidencePack:
    """Tests for finalize_evidence_pack node."""

    def test_finalize_basic(self, sample_plan, sample_candidates, sample_round_result):
        """Test basic evidence pack finalization."""
        state = {
            "plan": sample_plan,
            "evidence_pool": sample_candidates,
            "rounds": [sample_round_result],
            "retrieval_report": {},
        }

        result = finalize_evidence_pack(state)

        assert "final_evidence" in result
        assert "retrieval_report" in result

    def test_finalize_empty_pool(self, sample_plan):
        """Test finalization with empty evidence pool."""
        state = {
            "plan": sample_plan,
            "evidence_pool": [],
            "rounds": [],
            "retrieval_report": {},
        }

        result = finalize_evidence_pack(state)

        final = result["final_evidence"]
        assert final == []

    def test_finalize_respects_max_total_docs(self, sample_plan):
        """Test that finalization respects max_total_docs limit."""
        # Create 20 candidates
        pool = [
            Candidate(
                key=CandidateKey(f"doc{i}", "c1"),
                text=f"text{i}",
                rerank_score=0.9 - i * 0.01,
            )
            for i in range(20)
        ]

        # max_total_docs = 12 in sample_plan
        state = {
            "plan": sample_plan,
            "evidence_pool": pool,
            "rounds": [],
        }

        result = finalize_evidence_pack(state)

        final = result["final_evidence"]
        assert len(final) == 12

    def test_finalize_default_max_total_docs(self):
        """Test default max_total_docs when not specified."""
        pool = [
            Candidate(
                key=CandidateKey(f"doc{i}", "c1"),
                text=f"text{i}",
                rerank_score=0.9 - i * 0.01,
            )
            for i in range(20)
        ]

        state = {
            "plan": {},  # No stop_conditions
            "evidence_pool": pool,
            "rounds": [],
        }

        result = finalize_evidence_pack(state)

        # Should use DEFAULT_MAX_TOTAL_DOCS (12)
        final = result["final_evidence"]
        assert len(final) == 12

    def test_finalize_sorts_by_best_score(self):
        """Test that pool is sorted by best available score."""
        # Create candidates with different scores
        pool = [
            Candidate(key=CandidateKey("doc1", "c1"), text="t1", rerank_score=0.5),
            Candidate(key=CandidateKey("doc2", "c1"), text="t2", rerank_score=0.9),
            Candidate(key=CandidateKey("doc3", "c1"), text="t3", rerank_score=0.7),
        ]

        state = {
            "plan": {"stop_conditions": {"max_total_docs": 10}},
            "evidence_pool": pool,
            "rounds": [],
        }

        result = finalize_evidence_pack(state)

        final = result["final_evidence"]
        # Should be sorted by rerank_score descending
        assert final[0].rerank_score == 0.9
        assert final[1].rerank_score == 0.7
        assert final[2].rerank_score == 0.5

    def test_finalize_score_precedence(self):
        """Test score precedence: rerank > rrf > vector > bm25."""
        pool = [
            Candidate(key=CandidateKey("doc1", "c1"), text="t1", bm25_score=0.9),
            Candidate(key=CandidateKey("doc2", "c1"), text="t2", vector_score=0.85),
            Candidate(key=CandidateKey("doc3", "c1"), text="t3", rrf_score=0.88),
            Candidate(key=CandidateKey("doc4", "c1"), text="t4", rerank_score=0.95),
        ]

        state = {
            "plan": {"stop_conditions": {"max_total_docs": 10}},
            "evidence_pool": pool,
            "rounds": [],
        }

        result = finalize_evidence_pack(state)

        final = result["final_evidence"]
        # Should prefer rerank > rrf > vector > bm25
        assert final[0].key.doc_id == "doc4"  # rerank=0.95
        assert final[1].key.doc_id == "doc1"  # bm25=0.9 (highest among remaining)
        assert final[2].key.doc_id == "doc3"  # rrf=0.88
        assert final[3].key.doc_id == "doc2"  # vector=0.85

    def test_finalize_creates_report(self, sample_plan, sample_candidates):
        """Test that finalization creates detailed report."""
        rr1 = RoundResult(
            round_id=0,
            purpose="recall",
            queries=["q1"],
            raw_candidates_count=10,
            merged_candidates_count=8,
            reranked_candidates_count=5,
            selected=sample_candidates[:2],
            novelty_new_items=2,
        )

        rr2 = RoundResult(
            round_id=1,
            purpose="precision",
            queries=["q2", "q3"],
            raw_candidates_count=8,
            merged_candidates_count=6,
            reranked_candidates_count=4,
            selected=sample_candidates[2:],
            novelty_new_items=1,
        )

        state = {
            "plan": sample_plan,
            "evidence_pool": sample_candidates,
            "rounds": [rr1, rr2],
            "retrieval_report": {"skipped": False},
        }

        result = finalize_evidence_pack(state)

        report = result["retrieval_report"]
        assert report["round_count"] == 2
        assert report["final_docs"] == len(sample_candidates)
        assert "rounds" in report
        assert len(report["rounds"]) == 2

    def test_finalize_report_round_details(self, sample_plan, sample_candidates):
        """Test that report includes detailed round information."""
        rr = RoundResult(
            round_id=0,
            purpose="recall",
            queries=["query1", "query2"],
            raw_candidates_count=10,
            merged_candidates_count=8,
            reranked_candidates_count=5,
            selected=sample_candidates,
            novelty_new_items=3,
        )

        state = {
            "plan": sample_plan,
            "evidence_pool": sample_candidates,
            "rounds": [rr],
            "retrieval_report": {},
        }

        result = finalize_evidence_pack(state)

        report = result["retrieval_report"]
        round_info = report["rounds"][0]

        assert round_info["round_id"] == 0
        assert round_info["purpose"] == "recall"
        assert round_info["queries"] == ["query1", "query2"]
        assert round_info["raw_candidates_count"] == 10
        assert round_info["merged_candidates_count"] == 8
        assert round_info["reranked_candidates_count"] == 5
        assert round_info["novelty_new_items"] == 3
        assert "selected" in round_info

    def test_finalize_report_selected_format(self, sample_plan, sample_candidate):
        """Test that selected candidates are formatted correctly in report."""
        # Add scores to candidate
        candidate_with_scores = Candidate(
            key=CandidateKey("doc1", "chunk1"),
            text="test",
            rerank_score=0.95,
            rrf_score=0.88,
        )

        rr = RoundResult(
            round_id=0,
            purpose="recall",
            queries=["q1"],
            selected=[candidate_with_scores],
        )

        state = {
            "plan": sample_plan,
            "evidence_pool": [candidate_with_scores],
            "rounds": [rr],
        }

        result = finalize_evidence_pack(state)

        report = result["retrieval_report"]
        selected_info = report["rounds"][0]["selected"][0]

        assert selected_info["doc_id"] == "doc1"
        assert selected_info["chunk_id"] == "chunk1"
        assert selected_info["rerank_score"] == 0.95
        assert selected_info["rrf_score"] == 0.88

    def test_finalize_preserves_existing_report_fields(self, sample_plan, sample_candidates):
        """Test that existing report fields are preserved."""
        state = {
            "plan": sample_plan,
            "evidence_pool": sample_candidates,
            "rounds": [],
            "retrieval_report": {
                "skipped": False,
                "no_new_streak": 0,
                "custom_field": "custom_value",
            },
        }

        result = finalize_evidence_pack(state)

        report = result["retrieval_report"]
        # Should preserve existing fields
        assert report["skipped"] is False
        assert report["no_new_streak"] == 0
        assert report["custom_field"] == "custom_value"
        # And add new fields
        assert "round_count" in report
        assert "final_docs" in report

    def test_finalize_fewer_candidates_than_limit(self, sample_plan, sample_candidates):
        """Test when pool has fewer candidates than max_total_docs."""
        state = {
            "plan": sample_plan,  # max_total_docs = 12
            "evidence_pool": sample_candidates,  # Only 3 candidates
            "rounds": [],
        }

        result = finalize_evidence_pack(state)

        final = result["final_evidence"]
        # Should return all available candidates
        assert len(final) == len(sample_candidates)
