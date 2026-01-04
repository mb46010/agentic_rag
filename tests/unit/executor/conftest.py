# tests/unit/executor/conftest.py
"""Shared fixtures for executor module unit tests."""

import os
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentic_rag.executor.state import Candidate, CandidateKey, RoundResult

# Disable Langfuse for unit tests
os.environ["LANGFUSE_ENABLED"] = "0"


@pytest.fixture
def mock_retriever():
    """Mock RetrieverAdapter."""
    retriever = MagicMock()
    retriever.search = MagicMock(return_value=[])
    return retriever


@pytest.fixture
def mock_fusion():
    """Mock FusionAdapter."""
    fusion = MagicMock()
    fusion.rrf = MagicMock(return_value=[])
    return fusion


@pytest.fixture
def mock_reranker():
    """Mock RerankerAdapter."""
    reranker = MagicMock()
    reranker.rerank = MagicMock(return_value=[])
    return reranker


@pytest.fixture
def mock_hyde():
    """Mock HyDEAdapter."""
    hyde = MagicMock()
    hyde.synthesize = MagicMock(return_value="synthetic answer")
    hyde.derive_queries = MagicMock(return_value=["query1", "query2"])
    return hyde


@pytest.fixture
def mock_grader():
    """Mock CoverageGraderAdapter."""
    grader = MagicMock()
    grader.grade = MagicMock(return_value={
        "covered_entities": [],
        "missing_entities": [],
        "covered_subquestions": [],
        "missing_subquestions": [],
        "evidence_quality": "low",
        "confidence": 0.0,
        "contradictions": [],
    })
    return grader


@pytest.fixture
def sample_candidate():
    """Sample Candidate object."""
    return Candidate(
        key=CandidateKey(doc_id="doc1", chunk_id="chunk1"),
        text="This is sample text about Azure OpenAI.",
        metadata={"source": "docs.microsoft.com", "title": "Azure OpenAI Guide"},
        bm25_score=0.8,
        bm25_rank=1,
        vector_score=0.9,
        vector_rank=1,
    )


@pytest.fixture
def sample_candidates(sample_candidate):
    """List of sample Candidates."""
    candidates = [
        sample_candidate,
        Candidate(
            key=CandidateKey(doc_id="doc2", chunk_id="chunk1"),
            text="Another text about configuration.",
            metadata={"source": "internal", "title": "Config Guide"},
            bm25_score=0.7,
            vector_score=0.8,
        ),
        Candidate(
            key=CandidateKey(doc_id="doc1", chunk_id="chunk2"),
            text="More information from the same document.",
            metadata={"source": "docs.microsoft.com"},
            bm25_score=0.6,
            vector_score=0.7,
        ),
    ]
    return candidates


@pytest.fixture
def sample_plan():
    """Sample PlannerState (plan)."""
    return {
        "goal": "Find Azure OpenAI configuration",
        "strategy": "retrieve_then_answer",
        "retrieval_rounds": [
            {
                "round_id": 0,
                "purpose": "recall",
                "query_variants": ["Azure OpenAI configuration"],
                "retrieval_modes": [{"type": "hybrid", "k": 20, "alpha": 0.5}],
                "filters": {},
                "use_hyde": False,
                "rrf": True,
                "rerank": {"enabled": True, "model": "cross_encoder", "rerank_top_k": 60},
                "output": {"max_docs": 8},
            }
        ],
        "literal_constraints": {"must_preserve_terms": [], "must_match_exactly": False},
        "acceptance_criteria": {
            "min_independent_sources": 1,
            "must_cover_entities": ["Azure OpenAI"],
        },
        "stop_conditions": {
            "max_rounds": 1,
            "max_total_docs": 12,
            "confidence_threshold": 0.8,
            "no_new_information_rounds": 1,
        },
        "answer_requirements": {},
        "budget": {},
        "planner_meta": {},
    }


@pytest.fixture
def sample_executor_state(sample_plan):
    """Sample ExecutorState."""
    return {
        "plan": sample_plan,
        "normalized_query": "Azure OpenAI configuration",
        "constraints": {"domain": ["azure"]},
        "guardrails": {"sensitivity": "normal"},
        "signals": {"entities": [{"text": "Azure OpenAI", "type": "product"}]},
        "current_round_index": 0,
        "continue_search": True,
        "evidence_pool": [],
        "rounds": [],
        "errors": [],
    }


@pytest.fixture
def sample_round_result():
    """Sample RoundResult."""
    return RoundResult(
        round_id=0,
        purpose="recall",
        queries=["test query"],
        raw_candidates_count=10,
        merged_candidates_count=8,
        reranked_candidates_count=5,
        selected=[],
        novelty_new_items=5,
    )
