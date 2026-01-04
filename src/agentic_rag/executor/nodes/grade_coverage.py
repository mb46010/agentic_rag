# src/agentic_rag/executor/nodes/grade_coverage.py

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agentic_rag.executor.adapters import CoverageGraderAdapter
from agentic_rag.executor.state import Candidate, ExecutorState
from agentic_rag.executor.utils import observe, with_error_handling

logger = logging.getLogger(__name__)


def make_grade_coverage_node(grader: CoverageGraderAdapter):
    @observe
    @with_error_handling("grade_coverage")
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

        logger.info(f"Coverage: confidence={coverage.get('confidence', 0.0):.2f}, quality={coverage.get('evidence_quality', 'unknown')}")

        return {"coverage": coverage}

    return grade_coverage
