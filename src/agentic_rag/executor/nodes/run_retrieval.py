# src/agentic_rag/executor/nodes/run_retrieval.py

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from agentic_rag.executor.adapters import RetrieverAdapter
from agentic_rag.executor.constants import DEFAULT_RETRIEVAL_K
from agentic_rag.executor.state import Candidate, ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def make_run_retrieval_node(retriever: RetrieverAdapter):
    @observe
    @with_error_handling("run_retrieval")
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
        modes = round_spec.get("retrieval_modes") or [{"type": "hybrid", "k": DEFAULT_RETRIEVAL_K, "alpha": None}]

        raw: List[Candidate] = []

        for q in queries:
            for mode_spec in modes:
                mode = mode_spec.get("type", "hybrid")
                k = int(mode_spec.get("k", DEFAULT_RETRIEVAL_K))
                alpha: Optional[float] = mode_spec.get("alpha", None)

                # Adapter returns Candidates; we add provenance fields using replace() since Candidate is frozen
                hits = retriever.search(query=q, mode=mode, k=k, alpha=alpha, filters=filters)
                for h in hits:
                    enriched = replace(
                        h,
                        round_id=int(round_spec.get("round_id", idx)),
                        query=q,
                        mode=mode,
                    )
                    raw.append(enriched)

        logger.info(f"Retrieved {len(raw)} candidates across {len(queries)} queries and {len(modes)} modes")

        return {"round_candidates_raw": raw}

    return run_retrieval
