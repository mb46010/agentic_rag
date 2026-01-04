# src/agentic_rag/executor/nodes/select_evidence.py

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

from agentic_rag.executor.constants import DEFAULT_MAX_DOCS_PER_ROUND
from agentic_rag.executor.state import Candidate, ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def _diverse_top_k(cands: List[Candidate], max_docs: int) -> List[Candidate]:
    # Simple diversity: cap chunks per doc_id to reduce redundancy.
    by_doc: Dict[str, List[Candidate]] = defaultdict(list)
    for c in cands:
        by_doc[c.key.doc_id].append(c)

    # Sort within each doc by rerank_score/rrf_score
    for doc_id, lst in by_doc.items():
        lst.sort(key=lambda x: (x.rerank_score or x.rrf_score or 0.0), reverse=True)

    selected: List[Candidate] = []
    # Round-robin over docs
    while len(selected) < max_docs:
        progress = False
        for doc_id, lst in by_doc.items():
            if not lst:
                continue
            selected.append(lst.pop(0))
            progress = True
            if len(selected) >= max_docs:
                break
        if not progress:
            break
    return selected


@observe
@with_error_handling("select_evidence")
def select_evidence(state: ExecutorState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    rounds = plan.get("retrieval_rounds") or []
    idx = int(state.get("current_round_index", 0))
    round_spec = rounds[idx]

    reranked: List[Candidate] = list(state.get("round_candidates_reranked") or [])
    if not reranked:
        return {"round_selected": []}

    out_spec = round_spec.get("output") or {}
    max_docs = int(out_spec.get("max_docs", DEFAULT_MAX_DOCS_PER_ROUND))

    # Sort by best available score
    reranked.sort(key=lambda c: (c.rerank_score or c.rrf_score or 0.0), reverse=True)
    selected = _diverse_top_k(reranked, max_docs=max_docs)

    logger.info(f"Selected {len(selected)} evidence chunks from {len(reranked)} candidates")

    return {"round_selected": selected}
