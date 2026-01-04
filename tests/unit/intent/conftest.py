# tests/unit/intent/conftest.py
"""Shared fixtures for intent module unit tests."""

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
def sample_state(sample_messages):
    """Sample IntakeState for testing."""
    return {
        "messages": sample_messages,
        "user_context_info": None,
        "conversation_summary": None,
    }


@pytest.fixture
def mock_normalize_output():
    """Mock output from normalize_gate node."""
    return {
        "normalized_query": "configure Azure OpenAI for production",
        "constraints": {
            "domain": ["azure"],
            "format": [],
            "prohibitions": [],
            "nonfunctional": [],
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
        "language": "en",
        "locale": "en-US",
    }


@pytest.fixture
def mock_extract_signals_output():
    """Mock output from extract_signals node."""
    return {
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
    }
