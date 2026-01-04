# src/agentic_rag/executor/nodes/executor_gate.py

from __future__ import annotations

from typing import Any, Dict

from agentic_rag.executor.state import ExecutorState


def executor_gate(state: ExecutorState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    strategy = plan.get("strategy")

    if strategy in ("direct_answer", "clarify_then_retrieve", "defer_or_refuse"):
        # Executor does not run retrieval in these strategies.
        return {
            "continue_search": False,
            "final_evidence": [],
            "coverage": {
                "covered_entities": [],
                "missing_entities": [],
                "covered_subquestions": [],
                "missing_subquestions": [],
                "evidence_quality": "low",
                "confidence": 0.0,
                "contradictions": [],
            },
            "retrieval_report": {
                "skipped": True,
                "reason": f"strategy={strategy}",
            },
        }

    retrieval_rounds = plan.get("retrieval_rounds") or []
    if not retrieval_rounds:
        return {
            "errors": [
                {
                    "node": "executor_gate",
                    "type": "schema_validation",
                    "message": "Missing retrieval_rounds for retrieve_then_answer strategy.",
                    "retryable": False,
                    "details": None,
                }
            ],
            "continue_search": False,
        }

    stop_conditions = plan.get("stop_conditions") or {}
    max_rounds = stop_conditions.get("max_rounds", len(retrieval_rounds))
    max_rounds = max(1, int(max_rounds))

    # Minimal execution context placeholder (choose index/namespace later via adapter)
    execution_context = {
        "max_rounds": max_rounds,
        "max_total_docs": int(stop_conditions.get("max_total_docs", 12)),
        "confidence_threshold": stop_conditions.get("confidence_threshold", None),
        "no_new_information_rounds": int(stop_conditions.get("no_new_information_rounds", 1)),
    }

    return {
        "execution_context": execution_context,
        "current_round_index": 0,
        "rounds": [],
        "evidence_pool": [],
        "continue_search": True,
        "retrieval_report": {"skipped": False},
    }
