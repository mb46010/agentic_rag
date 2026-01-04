# src/agentic_rag/executor/nodes/rerank_candidates.py

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agentic_rag.executor.adapters import RerankerAdapter
from agentic_rag.executor.constants import DEFAULT_RERANK_TOP_K
from agentic_rag.executor.state import Candidate, ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def make_rerank_candidates_node(reranker: RerankerAdapter):
    @observe
    @with_error_handling("rerank_candidates")
    def rerank_candidates(state: ExecutorState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        rounds = plan.get("retrieval_rounds") or []
        idx = int(state.get("current_round_index", 0))
        round_spec = rounds[idx]

        merged: List[Candidate] = list(state.get("round_candidates_merged") or [])
        if not merged:
            return {"round_candidates_reranked": []}

        rerank_spec = round_spec.get("rerank") or {}
        enabled = bool(rerank_spec.get("enabled", True))
        top_k = int(rerank_spec.get("rerank_top_k", DEFAULT_RERANK_TOP_K))

        if not enabled:
            return {"round_candidates_reranked": merged}

        query = state.get("normalized_query", "")
        reranked = reranker.rerank(query=query, candidates=merged, top_k=top_k, context={"plan": plan})

        logger.info(f"Reranked {len(merged)} candidates to top {min(top_k, len(reranked))}")

        return {"round_candidates_reranked": reranked}

    return rerank_candidates
