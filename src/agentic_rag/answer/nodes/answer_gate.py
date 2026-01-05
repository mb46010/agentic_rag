# src/agentic_rag/answer/nodes/answer_gate.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from agentic_rag.answer.state import AnswerMode, AnswerState, CoverageModel

# Optional langfuse decorator - safe when disabled
try:
    from langfuse import observe  # type: ignore
except Exception:  # pragma: no cover

    def observe(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


def _has_no_code(constraints: Dict[str, Any]) -> bool:
    fmt = (constraints or {}).get("format") or []
    return "no_code" in fmt


def _get_sensitivity(guardrails: Dict[str, Any], plan: Dict[str, Any]) -> str:
    # Prefer plan safety if present, else intake guardrails
    plan_sens = ((plan or {}).get("safety") or {}).get("sensitivity")
    if plan_sens:
        return plan_sens
    return (guardrails or {}).get("sensitivity", "normal")


def _coverage_needs_clarification(coverage: CoverageModel, plan: Dict[str, Any]) -> bool:
    # Conservative: if coverage has missing items, or contradictions, or confidence too low.
    stop = (plan or {}).get("stop_conditions") or {}
    threshold = stop.get("confidence_threshold", None)

    if coverage.contradictions:
        return True
    if coverage.missing:
        return True
    if isinstance(threshold, (int, float)) and coverage.confidence < float(threshold):
        return True
    return False


def make_answer_gate_node():
    """Determine answer_mode: answer vs clarify vs refuse
    - refuse if sensitivity restricted (or plan strategy defer_or_refuse)
    - clarify if strategy clarify_then_retrieve OR evidence coverage is insufficient
    - answer otherwise
    """

    @observe
    def answer_gate(state: AnswerState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        guardrails = state.get("guardrails") or {}
        constraints = state.get("constraints") or {}

        strategy = plan.get("strategy")
        sensitivity = _get_sensitivity(guardrails, plan)

        # Hard refusal path
        if strategy == "defer_or_refuse" or sensitivity == "restricted":
            return {
                "answer_mode": "refuse",
                "answer_meta": {
                    "answer_version": "answer_v1",
                    "mode": "refuse",
                    "used_evidence_ids": [],
                    "coverage_confidence": 0.0,
                    "refusal": True,
                    "asked_clarification": False,
                },
            }

        # If planner explicitly said clarify-first
        if strategy == "clarify_then_retrieve":
            return {
                "answer_mode": "clarify",
                "answer_meta": {
                    "answer_version": "answer_v1",
                    "mode": "clarify",
                    "used_evidence_ids": [],
                    "coverage_confidence": 0.0,
                    "refusal": False,
                    "asked_clarification": True,
                },
            }

        # If retrieval strategy, check evidence coverage
        cov_raw = state.get("coverage") or {}
        try:
            cov = CoverageModel.model_validate(cov_raw)
        except Exception:
            cov = CoverageModel(confidence=0.0, covered=[], missing=[], contradictions=[])

        if _coverage_needs_clarification(cov, plan):
            return {
                "answer_mode": "clarify",
                "answer_meta": {
                    "answer_version": "answer_v1",
                    "mode": "clarify",
                    "used_evidence_ids": [],
                    "coverage_confidence": float(cov.confidence),
                    "refusal": False,
                    "asked_clarification": True,
                },
            }

        return {
            "answer_mode": "answer",
            "answer_meta": {
                "answer_version": "answer_v1",
                "mode": "answer",
                "used_evidence_ids": [],
                "coverage_confidence": float(cov.confidence),
                "refusal": False,
                "asked_clarification": False,
            },
        }

    return answer_gate
