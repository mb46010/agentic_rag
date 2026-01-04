# src/agentic_rag/executor/nodes/merge_candidates.py

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from agentic_rag.executor.adapters import FusionAdapter
from agentic_rag.executor.constants import DEFAULT_RRF_K, DEFAULT_RRF_POOL_SIZE
from agentic_rag.executor.state import Candidate, ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def _dedupe(cands: List[Candidate]) -> List[Candidate]:
    best: Dict[Any, Candidate] = {}
    for c in cands:
        key = c.key
        # Keep the candidate with best known rerank_score, else rrf_score, else any
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        prev_score = prev.rerank_score or prev.rrf_score or prev.vector_score or prev.bm25_score or 0.0
        cur_score = c.rerank_score or c.rrf_score or c.vector_score or c.bm25_score or 0.0
        if cur_score >= prev_score:
            best[key] = c
    return list(best.values())


def make_merge_candidates_node(fusion: FusionAdapter):
    @observe
    @with_error_handling("merge_candidates")
    def merge_candidates(state: ExecutorState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        rounds = plan.get("retrieval_rounds") or []
        idx = int(state.get("current_round_index", 0))
        round_spec = rounds[idx]

        raw: List[Candidate] = list(state.get("round_candidates_raw") or [])
        if not raw:
            return {"round_candidates_merged": []}

        # Group candidates by (query, mode) to build ranked lists for RRF
        by_list: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
        for c in raw:
            q = c.query or ""
            m = c.mode or "unknown"
            by_list[(q, m)].append(c)

        # Sort each list by a heuristic score if provided; otherwise preserve insertion
        ranked_lists: List[List[Candidate]] = []
        for _, lst in by_list.items():
            # Prefer explicit scores if present
            lst.sort(
                key=lambda x: (
                    (x.bm25_score or 0.0) + (x.vector_score or 0.0),
                    -(x.bm25_rank or 10**9),
                    -(x.vector_rank or 10**9),
                ),
                reverse=True,
            )
            ranked_lists.append(lst)

        use_rrf = bool(round_spec.get("rrf", True))
        if use_rrf and ranked_lists:
            fused = fusion.rrf(ranked_lists=ranked_lists, k=DEFAULT_RRF_POOL_SIZE, rrf_k=DEFAULT_RRF_K)
            merged = _dedupe(fused)
            merged.sort(key=lambda c: (c.rrf_score or 0.0), reverse=True)
        else:
            merged = _dedupe(raw)

        logger.info(f"Merged {len(raw)} raw candidates to {len(merged)} unique candidates, rrf={use_rrf}")

        return {"round_candidates_merged": merged}

    return merge_candidates
