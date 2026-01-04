# src/agentic_rag/executor/nodes/finalize_evidence_pack.py

from __future__ import annotations

from typing import Any, Dict, List

from agentic_rag.executor.state import Candidate, ExecutorState


def finalize_evidence_pack(state: ExecutorState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    pool: List[Candidate] = list(state.get("evidence_pool") or [])

    # Sort by best available score
    pool.sort(key=lambda c: (c.rerank_score or c.rrf_score or c.vector_score or c.bm25_score or 0.0), reverse=True)

    stop_conditions = plan.get("stop_conditions") or {}
    max_total_docs = int(stop_conditions.get("max_total_docs", 12))
    final = pool[:max_total_docs]

    # Build lightweight report
    rounds = state.get("rounds") or []
    report = state.get("retrieval_report") or {}
    report = {
        **report,
        "round_count": len(rounds),
        "final_docs": len(final),
        "rounds": [
            {
                "round_id": r.round_id,
                "purpose": r.purpose,
                "queries": r.queries,
                "raw_candidates_count": r.raw_candidates_count,
                "merged_candidates_count": r.merged_candidates_count,
                "reranked_candidates_count": r.reranked_candidates_count,
                "selected": [
                    {
                        "doc_id": c.key.doc_id,
                        "chunk_id": c.key.chunk_id,
                        "rerank_score": c.rerank_score,
                        "rrf_score": c.rrf_score,
                    }
                    for c in r.selected
                ],
                "novelty_new_items": r.novelty_new_items,
            }
            for r in rounds
        ],
    }

    return {"final_evidence": final, "retrieval_report": report}
