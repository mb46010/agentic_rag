# src/agentic_rag/executor/graph.py

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agentic_rag.executor.adapters import (
    CoverageGraderAdapter,
    FusionAdapter,
    HyDEAdapter,
    RerankerAdapter,
    RetrieverAdapter,
)
from agentic_rag.executor.nodes.executor_gate import executor_gate
from agentic_rag.executor.nodes.finalize_evidence_pack import finalize_evidence_pack
from agentic_rag.executor.nodes.grade_coverage import make_grade_coverage_node
from agentic_rag.executor.nodes.merge_candidates import make_merge_candidates_node
from agentic_rag.executor.nodes.prepare_round_queries import make_prepare_round_queries_node
from agentic_rag.executor.nodes.rerank_candidates import make_rerank_candidates_node
from agentic_rag.executor.nodes.run_retrieval import make_run_retrieval_node
from agentic_rag.executor.nodes.select_evidence import select_evidence
from agentic_rag.executor.nodes.should_continue import should_continue
from agentic_rag.executor.state import ExecutorState


def make_executor_graph(
    *,
    retriever: RetrieverAdapter,
    fusion: FusionAdapter,
    reranker: RerankerAdapter,
    hyde: HyDEAdapter,
    grader: CoverageGraderAdapter,
    max_retries: int = 2,
):
    retry_policy = RetryPolicy(max_attempts=max(1, int(max_retries)))

    g = StateGraph(ExecutorState)

    g.add_node("executor_gate", executor_gate, retry_policy=retry_policy)
    g.add_node("prepare_round_queries", make_prepare_round_queries_node(hyde), retry_policy=retry_policy)
    g.add_node("run_retrieval", make_run_retrieval_node(retriever), retry_policy=retry_policy)
    g.add_node("merge_candidates", make_merge_candidates_node(fusion), retry_policy=retry_policy)
    g.add_node("rerank_candidates", make_rerank_candidates_node(reranker), retry_policy=retry_policy)
    g.add_node("select_evidence", select_evidence, retry_policy=retry_policy)
    g.add_node("grade_coverage", make_grade_coverage_node(grader), retry_policy=retry_policy)
    g.add_node("should_continue", should_continue, retry_policy=retry_policy)
    g.add_node("finalize_evidence_pack", finalize_evidence_pack, retry_policy=retry_policy)

    g.add_edge(START, "executor_gate")

    # If executor_gate sets continue_search False, go finalize (empty evidence or skipped)
    def route_after_gate(state: ExecutorState):
        return "prepare_round_queries" if state.get("continue_search", False) else "finalize_evidence_pack"

    g.add_conditional_edges("executor_gate", route_after_gate, ["prepare_round_queries", "finalize_evidence_pack"])

    g.add_edge("prepare_round_queries", "run_retrieval")
    g.add_edge("run_retrieval", "merge_candidates")
    g.add_edge("merge_candidates", "rerank_candidates")
    g.add_edge("rerank_candidates", "select_evidence")
    g.add_edge("select_evidence", "grade_coverage")
    g.add_edge("grade_coverage", "should_continue")

    def route_loop(state: ExecutorState):
        return "prepare_round_queries" if state.get("continue_search", False) else "finalize_evidence_pack"

    g.add_conditional_edges("should_continue", route_loop, ["prepare_round_queries", "finalize_evidence_pack"])

    g.add_edge("finalize_evidence_pack", END)

    return g.compile()
