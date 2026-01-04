# tests/unit/executor/test_state.py
"""Unit tests for executor state models."""

import pytest
from dataclasses import FrozenInstanceError, replace

from agentic_rag.executor.state import (
    Candidate,
    CandidateKey,
    RoundResult,
    Coverage,
    ExecutorState,
    PlannerState,
    RetrievalRound,
    RetrievalModeSpec,
    RoundFilters,
    RerankSpec,
    RoundOutputSpec,
    LiteralConstraints,
    AcceptanceCriteria,
    StopConditions,
    AnswerRequirements,
    BudgetSpec,
    PlannerMeta,
)


class TestCandidateKey:
    """Tests for CandidateKey frozen dataclass."""

    def test_candidate_key_creation(self):
        """Test CandidateKey creation with required fields."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        assert key.doc_id == "doc1"
        assert key.chunk_id == "chunk1"

    def test_candidate_key_is_frozen(self):
        """Test that CandidateKey is immutable."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        with pytest.raises(FrozenInstanceError):
            key.doc_id = "doc2"

    def test_candidate_key_equality(self):
        """Test CandidateKey equality."""
        key1 = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        key2 = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        key3 = CandidateKey(doc_id="doc2", chunk_id="chunk1")

        assert key1 == key2
        assert key1 != key3

    def test_candidate_key_hashable(self):
        """Test that CandidateKey is hashable (can be used as dict key)."""
        key1 = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        key2 = CandidateKey(doc_id="doc2", chunk_id="chunk2")

        keys_dict = {key1: "value1", key2: "value2"}
        assert keys_dict[key1] == "value1"
        assert len(keys_dict) == 2


class TestCandidate:
    """Tests for Candidate frozen dataclass."""

    def test_candidate_minimal_creation(self):
        """Test Candidate creation with minimal required fields."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        candidate = Candidate(key=key, text="sample text")

        assert candidate.key == key
        assert candidate.text == "sample text"
        assert candidate.metadata == {}
        assert candidate.bm25_rank is None
        assert candidate.vector_rank is None
        assert candidate.bm25_score is None
        assert candidate.vector_score is None
        assert candidate.rrf_score is None
        assert candidate.rerank_score is None
        assert candidate.round_id is None
        assert candidate.query is None
        assert candidate.mode is None

    def test_candidate_full_creation(self):
        """Test Candidate creation with all fields."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        metadata = {"source": "test", "title": "Test Doc"}

        candidate = Candidate(
            key=key,
            text="full text",
            metadata=metadata,
            bm25_rank=1,
            vector_rank=2,
            bm25_score=0.8,
            vector_score=0.9,
            rrf_score=0.85,
            rerank_score=0.95,
            round_id=0,
            query="test query",
            mode="hybrid",
        )

        assert candidate.key == key
        assert candidate.text == "full text"
        assert candidate.metadata == metadata
        assert candidate.bm25_rank == 1
        assert candidate.vector_rank == 2
        assert candidate.bm25_score == 0.8
        assert candidate.vector_score == 0.9
        assert candidate.rrf_score == 0.85
        assert candidate.rerank_score == 0.95
        assert candidate.round_id == 0
        assert candidate.query == "test query"
        assert candidate.mode == "hybrid"

    def test_candidate_is_frozen(self):
        """Test that Candidate is immutable."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        candidate = Candidate(key=key, text="text")

        with pytest.raises(FrozenInstanceError):
            candidate.text = "new text"

        with pytest.raises(FrozenInstanceError):
            candidate.bm25_score = 0.5

    def test_candidate_use_replace(self):
        """Test using dataclasses.replace() to create modified copies."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        original = Candidate(key=key, text="original", bm25_score=0.8)

        # Create modified copy
        modified = replace(original, bm25_score=0.9, round_id=1)

        # Original unchanged
        assert original.bm25_score == 0.8
        assert original.round_id is None

        # Modified has new values
        assert modified.bm25_score == 0.9
        assert modified.round_id == 1
        assert modified.text == "original"  # Unchanged fields preserved

    def test_candidate_metadata_default_factory(self):
        """Test that metadata default factory creates independent dicts."""
        key1 = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        key2 = CandidateKey(doc_id="doc2", chunk_id="chunk2")

        c1 = Candidate(key=key1, text="text1")
        c2 = Candidate(key=key2, text="text2")

        # Metadata should be independent empty dicts
        assert c1.metadata == {}
        assert c2.metadata == {}
        assert c1.metadata is not c2.metadata


class TestRoundResult:
    """Tests for RoundResult dataclass."""

    def test_round_result_minimal_creation(self):
        """Test RoundResult with minimal fields."""
        result = RoundResult(round_id=0, purpose="recall")

        assert result.round_id == 0
        assert result.purpose == "recall"
        assert result.queries == []
        assert result.raw_candidates_count == 0
        assert result.merged_candidates_count == 0
        assert result.reranked_candidates_count == 0
        assert result.selected == []
        assert result.novelty_new_items == 0
        assert result.debug == {}

    def test_round_result_full_creation(self):
        """Test RoundResult with all fields."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        candidate = Candidate(key=key, text="text")

        result = RoundResult(
            round_id=1,
            purpose="precision",
            queries=["query1", "query2"],
            raw_candidates_count=20,
            merged_candidates_count=15,
            reranked_candidates_count=10,
            selected=[candidate],
            novelty_new_items=5,
            debug={"info": "test"},
        )

        assert result.round_id == 1
        assert result.purpose == "precision"
        assert result.queries == ["query1", "query2"]
        assert result.raw_candidates_count == 20
        assert result.merged_candidates_count == 15
        assert result.reranked_candidates_count == 10
        assert len(result.selected) == 1
        assert result.selected[0] == candidate
        assert result.novelty_new_items == 5
        assert result.debug == {"info": "test"}

    def test_round_result_is_mutable(self):
        """Test that RoundResult is mutable (not frozen)."""
        result = RoundResult(round_id=0, purpose="recall")

        # Should be able to modify fields
        result.raw_candidates_count = 10
        assert result.raw_candidates_count == 10

        result.queries.append("new query")
        assert "new query" in result.queries

    def test_round_result_default_factories(self):
        """Test that default factories create independent collections."""
        r1 = RoundResult(round_id=0, purpose="recall")
        r2 = RoundResult(round_id=1, purpose="precision")

        r1.queries.append("query1")
        r2.queries.append("query2")

        # Should be independent
        assert r1.queries == ["query1"]
        assert r2.queries == ["query2"]


class TestTypedDicts:
    """Tests for TypedDict structures."""

    def test_retrieval_mode_spec(self):
        """Test RetrievalModeSpec TypedDict."""
        spec: RetrievalModeSpec = {
            "type": "hybrid",
            "k": 20,
            "alpha": 0.5,
        }

        assert spec["type"] == "hybrid"
        assert spec["k"] == 20
        assert spec["alpha"] == 0.5

    def test_retrieval_mode_spec_partial(self):
        """Test RetrievalModeSpec with partial fields (total=False)."""
        spec: RetrievalModeSpec = {"type": "bm25"}
        assert spec["type"] == "bm25"
        assert "k" not in spec

    def test_round_filters(self):
        """Test RoundFilters TypedDict."""
        filters: RoundFilters = {
            "doc_types": ["documentation"],
            "domains": ["azure"],
            "entities": ["Azure OpenAI"],
            "time_range": "2023-2024",
        }

        assert filters["doc_types"] == ["documentation"]
        assert filters["domains"] == ["azure"]

    def test_rerank_spec(self):
        """Test RerankSpec TypedDict."""
        spec: RerankSpec = {
            "enabled": True,
            "model": "cross_encoder",
            "rerank_top_k": 60,
        }

        assert spec["enabled"] is True
        assert spec["model"] == "cross_encoder"
        assert spec["rerank_top_k"] == 60

    def test_round_output_spec(self):
        """Test RoundOutputSpec TypedDict."""
        spec: RoundOutputSpec = {"max_docs": 8}
        assert spec["max_docs"] == 8

    def test_retrieval_round(self):
        """Test RetrievalRound TypedDict."""
        round_spec: RetrievalRound = {
            "round_id": 0,
            "purpose": "recall",
            "query_variants": ["query1"],
            "retrieval_modes": [{"type": "hybrid", "k": 20, "alpha": 0.5}],
            "filters": {},
            "use_hyde": False,
            "rrf": True,
            "rerank": {"enabled": True, "model": "cross_encoder", "rerank_top_k": 60},
            "output": {"max_docs": 8},
        }

        assert round_spec["round_id"] == 0
        assert round_spec["purpose"] == "recall"
        assert len(round_spec["query_variants"]) == 1
        assert round_spec["rrf"] is True

    def test_literal_constraints(self):
        """Test LiteralConstraints TypedDict."""
        constraints: LiteralConstraints = {
            "must_preserve_terms": ["Azure", "OpenAI"],
            "must_match_exactly": True,
        }

        assert constraints["must_preserve_terms"] == ["Azure", "OpenAI"]
        assert constraints["must_match_exactly"] is True

    def test_acceptance_criteria(self):
        """Test AcceptanceCriteria TypedDict."""
        criteria: AcceptanceCriteria = {
            "min_independent_sources": 2,
            "require_authoritative_source": True,
            "must_cover_entities": ["Azure OpenAI"],
            "must_answer_subquestions": ["How to configure?"],
        }

        assert criteria["min_independent_sources"] == 2
        assert criteria["require_authoritative_source"] is True

    def test_stop_conditions(self):
        """Test StopConditions TypedDict."""
        conditions: StopConditions = {
            "max_rounds": 3,
            "max_total_docs": 12,
            "confidence_threshold": 0.8,
            "no_new_information_rounds": 2,
        }

        assert conditions["max_rounds"] == 3
        assert conditions["confidence_threshold"] == 0.8

    def test_answer_requirements(self):
        """Test AnswerRequirements TypedDict."""
        requirements: AnswerRequirements = {
            "format": ["markdown"],
            "tone": "professional",
            "length": "concise",
            "citation_style": "numbered",
        }

        assert requirements["format"] == ["markdown"]
        assert requirements["tone"] == "professional"

    def test_budget_spec(self):
        """Test BudgetSpec TypedDict."""
        budget: BudgetSpec = {
            "max_tokens": 4000,
            "max_latency_ms": 5000,
        }

        assert budget["max_tokens"] == 4000
        assert budget["max_latency_ms"] == 5000

    def test_planner_meta(self):
        """Test PlannerMeta TypedDict."""
        meta: PlannerMeta = {
            "planner_version": "v1",
            "rationale_tags": ["multi_step", "precision_needed"],
        }

        assert meta["planner_version"] == "v1"
        assert len(meta["rationale_tags"]) == 2

    def test_coverage(self):
        """Test Coverage TypedDict."""
        coverage: Coverage = {
            "covered_entities": ["Azure OpenAI"],
            "missing_entities": [],
            "covered_subquestions": ["How to configure?"],
            "missing_subquestions": [],
            "evidence_quality": "high",
            "confidence": 0.9,
            "contradictions": [],
        }

        assert coverage["covered_entities"] == ["Azure OpenAI"]
        assert coverage["evidence_quality"] == "high"
        assert coverage["confidence"] == 0.9


class TestPlannerState:
    """Tests for PlannerState TypedDict."""

    def test_planner_state_minimal(self):
        """Test PlannerState with minimal fields."""
        plan: PlannerState = {
            "goal": "Test goal",
            "strategy": "retrieve_then_answer",
        }

        assert plan["goal"] == "Test goal"
        assert plan["strategy"] == "retrieve_then_answer"

    def test_planner_state_full(self, sample_plan):
        """Test PlannerState with all fields (using fixture)."""
        assert sample_plan["goal"] == "Find Azure OpenAI configuration"
        assert sample_plan["strategy"] == "retrieve_then_answer"
        assert len(sample_plan["retrieval_rounds"]) == 1
        assert "stop_conditions" in sample_plan
        assert sample_plan["stop_conditions"]["max_rounds"] == 1

    def test_planner_state_strategies(self):
        """Test different strategy values."""
        strategies = [
            "direct_answer",
            "retrieve_then_answer",
            "clarify_then_retrieve",
            "defer_or_refuse",
        ]

        for strategy in strategies:
            plan: PlannerState = {"goal": "test", "strategy": strategy}
            assert plan["strategy"] == strategy


class TestExecutorState:
    """Tests for ExecutorState TypedDict."""

    def test_executor_state_minimal(self):
        """Test ExecutorState with minimal fields."""
        state: ExecutorState = {
            "plan": {"goal": "test", "strategy": "retrieve_then_answer"},
            "normalized_query": "test query",
        }

        assert state["plan"]["goal"] == "test"
        assert state["normalized_query"] == "test query"

    def test_executor_state_full(self, sample_executor_state):
        """Test ExecutorState with many fields (using fixture)."""
        assert sample_executor_state["normalized_query"] == "Azure OpenAI configuration"
        assert sample_executor_state["current_round_index"] == 0
        assert sample_executor_state["continue_search"] is True
        assert sample_executor_state["evidence_pool"] == []
        assert sample_executor_state["rounds"] == []
        assert "constraints" in sample_executor_state
        assert "guardrails" in sample_executor_state
        assert "signals" in sample_executor_state

    def test_executor_state_round_fields(self):
        """Test ExecutorState round-specific fields."""
        state: ExecutorState = {
            "plan": {"goal": "test", "strategy": "retrieve_then_answer"},
            "normalized_query": "test",
            "current_round_index": 1,
            "round_queries": ["query1", "query2"],
            "round_candidates_raw": [],
            "round_candidates_merged": [],
            "round_candidates_reranked": [],
            "round_selected": [],
        }

        assert state["current_round_index"] == 1
        assert state["round_queries"] == ["query1", "query2"]
        assert state["round_candidates_raw"] == []

    def test_executor_state_aggregation_fields(self):
        """Test ExecutorState aggregation fields."""
        key = CandidateKey(doc_id="doc1", chunk_id="chunk1")
        candidate = Candidate(key=key, text="text")
        round_result = RoundResult(round_id=0, purpose="recall", selected=[candidate])

        state: ExecutorState = {
            "plan": {"goal": "test", "strategy": "retrieve_then_answer"},
            "normalized_query": "test",
            "rounds": [round_result],
            "evidence_pool": [candidate],
            "final_evidence": [candidate],
            "coverage": {
                "covered_entities": ["test"],
                "missing_entities": [],
                "evidence_quality": "high",
                "confidence": 0.9,
            },
        }

        assert len(state["rounds"]) == 1
        assert len(state["evidence_pool"]) == 1
        assert state["coverage"]["confidence"] == 0.9

    def test_executor_state_control_flow(self):
        """Test ExecutorState control flow fields."""
        state: ExecutorState = {
            "plan": {"goal": "test", "strategy": "retrieve_then_answer"},
            "normalized_query": "test",
            "continue_search": False,
            "errors": [{"node": "test", "type": "error", "message": "test error"}],
        }

        assert state["continue_search"] is False
        assert len(state["errors"]) == 1
        assert state["errors"][0]["node"] == "test"
