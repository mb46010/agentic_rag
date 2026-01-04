# src/agentic_rag/executor/nodes/prepare_round_queries.py

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agentic_rag.executor.adapters import HyDEAdapter
from agentic_rag.executor.state import ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def _preserve_literal_terms(queries: List[str], must_preserve_terms: List[str]) -> List[str]:
    if not must_preserve_terms:
        return queries

    preserved = []
    for q in queries:
        ok = True
        for term in must_preserve_terms:
            if term and term not in q:
                ok = False
                break
        if ok:
            preserved.append(q)

    # Ensure at least one query that contains all terms (append a deterministic one if needed)
    if not preserved:
        base = queries[0] if queries else ""
        literal_tail = " ".join([t for t in must_preserve_terms if t])
        preserved = [f"{base} {literal_tail}".strip()]

    # Keep original queries too, but put preserved first
    merged = preserved + [q for q in queries if q not in preserved]
    return merged


def make_prepare_round_queries_node(hyde: HyDEAdapter):
    @observe
    @with_error_handling("prepare_round_queries")
    def prepare_round_queries(state: ExecutorState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        rounds = plan.get("retrieval_rounds") or []
        idx = int(state.get("current_round_index", 0))

        if idx >= len(rounds):
            return {"continue_search": False}

        round_spec = rounds[idx]
        base_variants = list(round_spec.get("query_variants") or [])
        if not base_variants:
            # Fallback to normalized_query
            base_variants = [state.get("normalized_query", "")]

        literal_constraints = plan.get("literal_constraints") or {}
        must_preserve = list(literal_constraints.get("must_preserve_terms") or [])
        use_hyde = bool(round_spec.get("use_hyde", False))

        queries = base_variants

        # HyDE: only if enabled and no strict literal constraints
        must_match_exactly = bool(literal_constraints.get("must_match_exactly", False))
        if use_hyde and not must_match_exactly and not must_preserve:
            synthetic = hyde.synthesize(query=state.get("normalized_query", ""), context={"plan": plan})
            derived = hyde.derive_queries(
                original_query=state.get("normalized_query", ""),
                synthetic_answer=synthetic,
                max_queries=4,
            )
            # Include original first
            queries = [state.get("normalized_query", "")] + derived

        queries = _preserve_literal_terms(queries, must_preserve)

        logger.info(f"Prepared {len(queries)} queries for round {idx}, use_hyde={use_hyde and not must_match_exactly}")
        logger.debug(f"Queries: {queries}")

        return {"round_queries": queries}

    return prepare_round_queries
