# src/agentic_rag/answer/graph.py
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agentic_rag.answer.nodes.answer_gate import make_answer_gate_node
from agentic_rag.answer.nodes.compose_answer import make_compose_answer_node
from agentic_rag.answer.nodes.postprocess_answer import make_postprocess_answer_node
from agentic_rag.answer.state import AnswerState


def make_answer_graph(llm, max_retries: int = 2):
    retry_policy = RetryPolicy(max_attempts=max(1, int(max_retries)))

    g = StateGraph(AnswerState)

    g.add_node("answer_gate", make_answer_gate_node())
    g.add_node("compose_answer", make_compose_answer_node(llm), retry=retry_policy)
    g.add_node("postprocess_answer", make_postprocess_answer_node())

    g.add_edge(START, "answer_gate")
    g.add_edge("answer_gate", "compose_answer")
    g.add_edge("compose_answer", "postprocess_answer")
    g.add_edge("postprocess_answer", END)

    return g.compile()
