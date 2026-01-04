# src/agentic_rag/intent/graph.py

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agentic_rag.intent.nodes.extract_signals import make_extract_signals_node
from agentic_rag.intent.nodes.normalize_gate import make_normalize_gate_node
from agentic_rag.intent.state import IntakeState

logger = logging.getLogger(__name__)


def make_intake_graph(llm, max_retries: int = 3):
    # Use the function argument, not a hardcoded constant.
    retry_policy = RetryPolicy(max_attempts=max_retries)

    # Provide the state schema (TypedDict) to StateGraph for correctness and tooling.
    intent_graph_builder = StateGraph(IntakeState)

    # Node factory signatures should be consistent: make_*_node(llm) -> callable
    intent_graph_builder.add_node(
        "normalize_gate",
        make_normalize_gate_node(llm),
        retry=retry_policy,
    )

    intent_graph_builder.add_node(
        "extract_signals",
        make_extract_signals_node(llm),
        retry=retry_policy,
    )

    intent_graph_builder.add_edge(START, "normalize_gate")
    intent_graph_builder.add_edge("normalize_gate", "extract_signals")
    intent_graph_builder.add_edge("extract_signals", END)

    return intent_graph_builder.compile()
