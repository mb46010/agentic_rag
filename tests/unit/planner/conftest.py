# tests/unit/planner/conftest.py
"""Shared fixtures for planner module unit tests."""

import os
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# Disable Langfuse for unit tests
os.environ["LANGFUSE_ENABLED"] = "0"


@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable structured output."""
    llm = MagicMock()

    # Mock with_structured_output to return a mock that can invoke
    def default_with_structured_output(schema, **kwargs):
        mock_chain = MagicMock()
        mock_chain._schema = schema
        return mock_chain

    llm.with_structured_output = MagicMock(side_effect=default_with_structured_output)

    return llm


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [{"role": "user", "content": "How do I configure Azure OpenAI for production?"}]


@pytest.fixture
def sample_intake_state(sample_messages):
    """Sample IntakeState for planner testing."""
    return {
        "messages": sample_messages,
        "normalized_query": "configure Azure OpenAI for production",
        "constraints": {
            "domain": ["azure"],
            "format": [],
            "prohibitions": [],
            "nonfunctional": ["high_precision"],
        },
        "guardrails": {
            "time_sensitivity": "low",
            "context_dependency": "weak",
            "sensitivity": "normal",
            "pii_present": False,
        },
        "clarification": {
            "needed": False,
            "blocking": False,
            "reasons": [],
        },
        "user_intent": "plan",
        "retrieval_intent": "procedure",
        "answerability": "internal_corpus",
        "complexity_flags": ["requires_synthesis"],
        "signals": {
            "entities": [{"text": "Azure OpenAI", "type": "product", "confidence": "high"}],
            "acronyms": [],
            "artifact_flags": [],
            "literal_terms": [],
        },
        "language": "en",
        "locale": "en-US",
    }


@pytest.fixture
def sample_planner_output():
    """Sample valid PlannerState output."""
    return {
        "goal": "configure Azure OpenAI for production",
        "strategy": "retrieve_then_answer",
        "clarifying_questions": [],
        "retrieval_rounds": [
            {
                "round_id": 0,
                "purpose": "recall",
                "query_variants": ["Azure OpenAI production configuration"],
                "retrieval_modes": [
                    {
                        "type": "hybrid",
                        "k": 20,
                        "alpha": 0.5,
                    }
                ],
                "filters": {},
                "use_hyde": False,
                "rrf": True,
                "rerank": {
                    "enabled": True,
                    "model": "cross_encoder",
                    "rerank_top_k": 60,
                },
                "output": {
                    "max_docs": 8,
                },
            }
        ],
        "literal_constraints": {
            "must_preserve_terms": [],
            "must_match_exactly": False,
        },
        "acceptance_criteria": {
            "min_independent_sources": 1,
            "require_authoritative_source": False,
            "must_cover_entities": [],
            "must_answer_subquestions": [],
        },
        "stop_conditions": {
            "max_rounds": 2,
            "max_total_docs": 12,
            "confidence_threshold": None,
            "no_new_information_rounds": 1,
        },
        "answer_requirements": {
            "format": [],
            "tone": None,
            "length": None,
            "citation_style": None,
        },
        "budget": {
            "max_tokens": 8000,
            "max_latency_ms": None,
        },
        "safety": {
            "sensitivity": "normal",
            "pii_allowed": False,
        },
        "planner_meta": {
            "planner_version": "planner_v1",
            "rationale_tags": [],
        },
    }
