# tests/unit/answer/conftest.py
"""Shared fixtures for answer module unit tests."""

import os
from unittest.mock import MagicMock

import pytest

# Disable Langfuse for unit tests
os.environ["LANGFUSE_ENABLED"] = "0"


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    # Create a mock chain that's returned by with_structured_output
    mock_chain = MagicMock()
    llm.with_structured_output = MagicMock(return_value=mock_chain)
    llm.__or__ = MagicMock(return_value=mock_chain)  # For prompt | model
    mock_chain.invoke = MagicMock()
    llm._mock_chain = mock_chain  # Store reference for test access
    return llm


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"type": "human", "content": "What is LangGraph?"},
    ]


@pytest.fixture
def sample_plan():
    """Sample plan dict."""
    return {
        "goal": "Explain LangGraph",
        "strategy": "retrieve_then_answer",
        "retrieval_rounds": [],
        "stop_conditions": {"confidence_threshold": 0.7},
    }


@pytest.fixture
def sample_constraints():
    """Sample constraints dict."""
    return {
        "domain": ["langgraph"],
        "format": [],
        "prohibitions": [],
    }


@pytest.fixture
def sample_guardrails():
    """Sample guardrails dict."""
    return {
        "time_sensitivity": "none",
        "context_dependency": "none",
        "sensitivity": "normal",
        "pii_present": False,
    }


@pytest.fixture
def sample_coverage():
    """Sample coverage dict."""
    return {
        "confidence": 0.8,
        "covered": ["LangGraph"],
        "missing": [],
        "contradictions": [],
    }


@pytest.fixture
def sample_evidence():
    """Sample evidence list."""
    return [
        {
            "evidence_id": "ev_001",
            "text": "LangGraph is a library for building stateful applications.",
            "source": "docs.langgraph.com",
            "metadata": {"title": "LangGraph Introduction"},
        },
        {
            "evidence_id": "ev_002",
            "text": "LangGraph uses StateGraph to define workflows.",
            "source": "docs.langgraph.com",
            "metadata": {"title": "StateGraph Guide"},
        },
    ]


@pytest.fixture
def sample_answer_state(
    sample_messages,
    sample_plan,
    sample_constraints,
    sample_guardrails,
    sample_coverage,
    sample_evidence,
):
    """Sample AnswerState."""
    return {
        "messages": sample_messages,
        "plan": sample_plan,
        "constraints": sample_constraints,
        "guardrails": sample_guardrails,
        "coverage": sample_coverage,
        "final_evidence": sample_evidence,
        "normalized_query": "Explain what LangGraph is",
        "language": "en",
        "locale": "en-US",
    }
