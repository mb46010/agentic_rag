from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agentic_rag.intent.state import IntakeState
from agentic_rag.planner.nodes.planner import make_planner_node


def make_planner_graph(llm, max_retries: int = 2):
    retry_policy = RetryPolicy(max_attempts=max(1, int(max_retries)))

    g = StateGraph(IntakeState)
    g.add_node("planner", make_planner_node(llm), retry_policy=retry_policy)
    g.add_edge(START, "planner")
    g.add_edge("planner", END)
    return g.compile()
