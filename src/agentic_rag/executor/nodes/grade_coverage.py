# src/agentic_rag/executor/nodes/grade_coverage.py

from __future__ import annotations

from typing import Any, Dict, List

from agentic_rag.executor.adapters import CoverageGraderAdapter
from agentic_rag.executor.state import Candidate, ExecutorState


def make_grade_coverage_node(grader: CoverageGraderAdapter):
    def grade_coverage(state: ExecutorState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        selected: List[Candidate] = list(state.get("round_selected") or [])

        coverage = grader.grade(
            plan=plan,
            normalized_query=state.get("normalized_query", ""),
            selected_evidence=selected,
            context={
                "constraints": state.get("constraints") or {},
                "guardrails": state.get("guardrails") or {},
            },
        )
        return {"coverage": coverage}

    return grade_coverage
