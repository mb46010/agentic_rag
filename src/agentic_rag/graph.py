# src/agentic_rag/graph.py
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.answer.graph import make_answer_graph
from agentic_rag.executor.adapters import (
    CoverageGraderAdapter,
    FusionAdapter,
    HyDEAdapter,
    RerankerAdapter,
    RetrieverAdapter,
)
from agentic_rag.executor.graph import make_executor_graph
from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.state import AgentState


def make_agent_graph(
    llm,
    *,
    retriever: RetrieverAdapter,
    fusion: FusionAdapter,
    reranker: RerankerAdapter,
    hyde: HyDEAdapter,
    grader: CoverageGraderAdapter,
    max_retries: int = 2,
):
    """Create the Master Agent Graph."""
    # 1. compile subgraphs
    intake = make_intake_graph(llm, max_retries=max_retries)
    planner = make_planner_graph(llm, max_retries=max_retries)
    executor = make_executor_graph(
        retriever=retriever,
        fusion=fusion,
        reranker=reranker,
        hyde=hyde,
        grader=grader,
        max_retries=max_retries,
    )
    answer = make_answer_graph(llm, max_retries=max_retries)

    # 2. construct master graph
    workflow = StateGraph(AgentState)

    workflow.add_node("intake", intake)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("answer", answer)

    # 3. define edges
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "planner")

    # Conditional logic could go here:
    # if planner says "direct_answer", skip executor?
    # For now, we trust the executor gate to handle "skip" if plan says so,
    # or we can route explicitly.
    # Let's check planner strategy.

    def route_after_planner(state: AgentState):
        plan = state.get("plan") or {}
        strategy = plan.get("strategy", "retrieve_then_answer")

        if strategy == "direct_answer":
            # If direct answer, we might still want strictly formatted output from answer module?
            # Or does planner give the answer?
            # Usually direct_answer means "LLM answer without retrieval".
            # So we skip executor and go to answer.
            return "answer"

        if strategy == "defer_or_refuse":
            # Skip executor, go to answer (which will refuse based on plan)
            return "answer"

        return "executor"

    workflow.add_conditional_edges("planner", route_after_planner, ["executor", "answer"])

    workflow.add_edge("executor", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()
