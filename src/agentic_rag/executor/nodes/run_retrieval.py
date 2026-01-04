# src/agentic_rag/executor/nodes/run_retrieval.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_rag.executor.adapters import RetrieverAdapter
from agentic_rag.executor.state import Candidate, ExecutorState


def make_run_retrieval_node(retriever: RetrieverAdapter):
    def run_retrieval(state: ExecutorState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        rounds = plan.get("retrieval_rounds") or []
        idx = int(state.get("current_round_index", 0))
        round_spec = rounds[idx]

        queries: List[str] = list(state.get("round_queries") or [])
        if not queries:
            return {
                "errors": [
                    {
                        "node": "run_retrieval",
                        "type": "schema_validation",
                        "message": "Missing round_queries",
                        "retryable": False,
                        "details": None,
                    }
                ]
            }

        filters = round_spec.get("filters") or {}
        modes = round_spec.get("retrieval_modes") or [{"type": "hybrid", "k": 20, "alpha": None}]

        raw: List[Candidate] = []

        for q in queries:
            for mode_spec in modes:
                mode = mode_spec.get("type", "hybrid")
                k = int(mode_spec.get("k", 20))
                alpha: Optional[float] = mode_spec.get("alpha", None)

                # Adapter returns Candidates; we add provenance fields.
                hits = retriever.search(query=q, mode=mode, k=k, alpha=alpha, filters=filters)
                for h in hits:
                    h.round_id = int(round_spec.get("round_id", idx))
                    h.query = q
                    h.mode = mode
                raw.extend(hits)

        return {"round_candidates_raw": raw}

    return run_retrieval
