# tests/unit/executor/test_should_continue.py
"""Unit tests for should_continue node."""

import pytest

from agentic_rag.executor.nodes.should_continue import should_continue
from agentic_rag.executor.state import Candidate, CandidateKey


class TestShouldContinue:
    """Tests for should_continue node."""

    def test_continue_basic(self, sample_plan, sample_candidates):
        """Test basic should_continue logic."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["query1"],
            "round_candidates_raw": sample_candidates,
            "round_candidates_merged": sample_candidates,
            "round_candidates_reranked": sample_candidates,
            "round_selected": sample_candidates[:2],
            "evidence_pool": [],
            "rounds": [],
            "coverage": {"confidence": 0.5},
        }

        result = should_continue(state)

        assert "continue_search" in result
        assert "evidence_pool" in result
        assert "rounds" in result
        assert "current_round_index" in result

    def test_continue_stops_at_max_rounds(self, sample_plan, sample_candidates):
        """Test that search stops when max_rounds reached."""
        # max_rounds = 1 in sample_plan
        state = {
            "plan": sample_plan,
            "current_round_index": 0,  # This is the last round (0-indexed)
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [],
        }

        result = should_continue(state)

        # Should stop (reached max_rounds)
        assert result["continue_search"] is False
        assert result["current_round_index"] == 0  # Doesn't increment when stopped

    def test_continue_when_more_rounds_available(self, sample_plan, sample_candidates):
        """Test that search continues when more rounds available."""
        plan = {**sample_plan}
        plan["stop_conditions"]["max_rounds"] = 3
        # Add more retrieval rounds
        plan["retrieval_rounds"] = [
            {"round_id": 0, "purpose": "recall"},
            {"round_id": 1, "purpose": "precision"},
            {"round_id": 2, "purpose": "verification"},
        ]

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [],
            "coverage": {"confidence": 0.5},
        }

        result = should_continue(state)

        # Should continue (not at max yet)
        assert result["continue_search"] is True
        assert result["current_round_index"] == 1  # Incremented

    def test_continue_stops_when_confidence_met(self, sample_plan, sample_candidates):
        """Test that search stops when confidence threshold met."""
        plan = {**sample_plan}
        plan["stop_conditions"]["max_rounds"] = 3
        plan["stop_conditions"]["confidence_threshold"] = 0.8

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [],
            "coverage": {"confidence": 0.9},  # Meets threshold
        }

        result = should_continue(state)

        # Should stop (confidence met)
        assert result["continue_search"] is False

    def test_continue_when_confidence_not_met(self, sample_plan, sample_candidates):
        """Test that search continues when confidence threshold not met."""
        plan = {**sample_plan}
        plan["stop_conditions"]["max_rounds"] = 3
        plan["stop_conditions"]["confidence_threshold"] = 0.8

        state = {
            "plan": plan,
            "current_round_index": 0,
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [],
            "coverage": {"confidence": 0.5},  # Below threshold
        }

        result = should_continue(state)

        # Should continue
        assert result["continue_search"] is True

    def test_continue_tracks_novelty(self, sample_plan):
        """Test that novelty (new items) is tracked correctly."""
        existing_pool = [
            Candidate(key=CandidateKey("doc1", "c1"), text="existing1"),
            Candidate(key=CandidateKey("doc2", "c1"), text="existing2"),
        ]

        new_selected = [
            Candidate(key=CandidateKey("doc1", "c1"), text="existing1"),  # Duplicate
            Candidate(key=CandidateKey("doc3", "c1"), text="new1"),  # New
            Candidate(key=CandidateKey("doc4", "c1"), text="new2"),  # New
        ]

        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["query1"],
            "round_candidates_raw": new_selected,
            "round_candidates_merged": new_selected,
            "round_candidates_reranked": new_selected,
            "round_selected": new_selected,
            "evidence_pool": existing_pool,
            "rounds": [],
        }

        result = should_continue(state)

        # Should have 2 new items
        rounds = result["rounds"]
        assert len(rounds) == 1
        assert rounds[0].novelty_new_items == 2

        # Pool should have 2 new items added
        assert len(result["evidence_pool"]) == 4

    def test_continue_stops_on_no_novelty_streak(self, sample_plan):
        """Test that search stops when no novelty streak exceeds limit."""
        plan = {**sample_plan}
        plan["stop_conditions"]["max_rounds"] = 5
        plan["stop_conditions"]["no_new_information_rounds"] = 2

        existing_pool = [Candidate(key=CandidateKey("doc1", "c1"), text="existing")]

        state = {
            "plan": plan,
            "current_round_index": 1,
            "round_selected": existing_pool,  # All duplicates
            "evidence_pool": existing_pool,
            "rounds": [],
            "retrieval_report": {"no_new_streak": 1},  # Already at 1
        }

        result = should_continue(state)

        # Should stop (no_new_streak would be 2, >= limit)
        assert result["continue_search"] is False
        assert result["retrieval_report"]["no_new_streak"] == 2

    def test_continue_resets_novelty_streak(self, sample_plan):
        """Test that novelty streak resets when new items found."""
        plan = {**sample_plan}
        plan["stop_conditions"]["max_rounds"] = 5
        plan["stop_conditions"]["no_new_information_rounds"] = 2

        existing_pool = [Candidate(key=CandidateKey("doc1", "c1"), text="existing")]
        new_selected = [Candidate(key=CandidateKey("doc2", "c1"), text="new")]

        state = {
            "plan": plan,
            "current_round_index": 1,
            "round_selected": new_selected,  # Has new item
            "evidence_pool": existing_pool,
            "rounds": [],
            "retrieval_report": {"no_new_streak": 1},  # Previous streak
        }

        result = should_continue(state)

        # Streak should reset to 0
        assert result["retrieval_report"]["no_new_streak"] == 0
        assert result["continue_search"] is True

    def test_continue_creates_round_result(self, sample_plan, sample_candidates):
        """Test that RoundResult is created correctly."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_queries": ["q1", "q2"],
            "round_candidates_raw": sample_candidates * 2,
            "round_candidates_merged": sample_candidates,
            "round_candidates_reranked": sample_candidates[:2],
            "round_selected": sample_candidates[:1],
            "evidence_pool": [],
            "rounds": [],
        }

        result = should_continue(state)

        rounds = result["rounds"]
        assert len(rounds) == 1

        rr = rounds[0]
        assert rr.round_id == 0
        assert rr.purpose == "recall"
        assert rr.queries == ["q1", "q2"]
        assert rr.raw_candidates_count == 6  # 3 * 2
        assert rr.merged_candidates_count == 3
        assert rr.reranked_candidates_count == 2
        assert len(rr.selected) == 1
        assert rr.novelty_new_items == 1

    def test_continue_appends_to_existing_rounds(self, sample_plan, sample_candidates, sample_round_result):
        """Test that round results are appended to existing list."""
        state = {
            "plan": sample_plan,
            "current_round_index": 1,
            "round_queries": ["q1"],
            "round_candidates_raw": sample_candidates,
            "round_candidates_merged": sample_candidates,
            "round_candidates_reranked": sample_candidates,
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [sample_round_result],  # Existing round
        }

        result = should_continue(state)

        rounds = result["rounds"]
        assert len(rounds) == 2  # Original + new

    def test_continue_handles_missing_coverage(self, sample_plan, sample_candidates):
        """Test handling when coverage is missing."""
        state = {
            "plan": sample_plan,
            "current_round_index": 0,
            "round_selected": sample_candidates,
            "evidence_pool": [],
            "rounds": [],
            # Missing coverage
        }

        result = should_continue(state)

        # Should use confidence = 0.0
        assert "continue_search" in result
