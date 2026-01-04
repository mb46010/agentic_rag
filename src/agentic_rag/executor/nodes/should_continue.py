# src/agentic_rag/executor/nodes/should_continue.py

from __future__ import annotations

from typing import Any, Dict, List

from agentic_rag.executor.state import Candidate, ExecutorState, RoundResult


def should_continue(state: ExecutorState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    rounds_spec = plan.get("retrieval_rounds") or []
    idx = int(state.get("current_round_index", 0))
    max_rounds = int((plan.get("stop_conditions") or {}).get("max_rounds", len(rounds_spec)))

    selected: List[Candidate] = list(state.get("round_selected") or [])
    evidence_pool: List[Candidate] = list(state.get("evidence_pool") or [])

    # Novelty: count new keys not in pool
    existing_keys = {c.key for c in evidence_pool}
    new_items = [c for c in selected if c.key not in existing_keys]
    novelty = len(new_items)

    # Update pool
    new_pool = evidence_pool + new_items

    # Capture round result for reporting
    rr = RoundResult(
        round_id=int((rounds_spec[idx].get("round_id", idx)) if idx < len(rounds_spec) else idx),
        purpose=str((rounds_spec[idx].get("purpose", "unknown")) if idx < len(rounds_spec) else "unknown"),
        queries=list(state.get("round_queries") or []),
        raw_candidates_count=len(state.get("round_candidates_raw") or []),
        merged_candidates_count=len(state.get("round_candidates_merged") or []),
        reranked_candidates_count=len(state.get("round_candidates_reranked") or []),
        selected=selected,
        novelty_new_items=novelty,
        debug={},
    )
    rounds_log = list(state.get("rounds") or [])
    rounds_log.append(rr)

    # Decide stopping
    stop_conditions = plan.get("stop_conditions") or {}
    threshold = stop_conditions.get("confidence_threshold", None)
    no_new_limit = int(stop_conditions.get("no_new_information_rounds", 1))

    coverage = state.get("coverage") or {}
    confidence = float(coverage.get("confidence", 0.0))

    # Track consecutive no-novelty rounds in retrieval_report
    report = state.get("retrieval_report") or {}
    no_new_streak = int(report.get("no_new_streak", 0))
    if novelty == 0:
        no_new_streak += 1
    else:
        no_new_streak = 0

    # Basic rules:
    # - stop if reached max rounds
    # - stop if confidence meets threshold (if provided)
    # - stop if no novelty streak exceeds limit
    reached_max = (idx + 1) >= min(max_rounds, len(rounds_spec))
    meets_conf = (threshold is not None) and (confidence >= float(threshold))
    stale = no_new_streak >= no_new_limit

    cont = not (reached_max or meets_conf or stale)

    return {
        "evidence_pool": new_pool,
        "rounds": rounds_log,
        "continue_search": cont,
        "retrieval_report": {**report, "no_new_streak": no_new_streak},
        "current_round_index": (idx + 1) if cont else idx,
    }
