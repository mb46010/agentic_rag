# src/agentic_rag/answer/nodes/postprocess_answer.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from agentic_rag.answer.state import AnswerState, EvidenceItem

# Optional langfuse decorator - safe when disabled
try:
    from langfuse import observe  # type: ignore
except Exception:  # pragma: no cover

    def observe(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _has_no_code(constraints: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    fmt = (constraints or {}).get("format") or []
    req_fmt = ((plan or {}).get("answer_requirements") or {}).get("format") or []
    return ("no_code" in fmt) or ("no_code" in req_fmt)


def _valid_evidence_ids(final_evidence: Any) -> Set[str]:
    ids: Set[str] = set()
    if not isinstance(final_evidence, list):
        return ids
    for x in final_evidence:
        try:
            ev = EvidenceItem.model_validate(x)
            ids.add(ev.evidence_id)
        except Exception:
            continue
    return ids


def make_postprocess_answer_node():
    @observe
    def postprocess_answer(state: AnswerState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        constraints = state.get("constraints") or {}
        answer = state.get("final_answer") or ""
        citations = state.get("citations") or []

        # Enforce no_code by removing fenced blocks (cheap deterministic safety)
        if _has_no_code(constraints, plan):
            answer = _CODE_FENCE_RE.sub("", answer).strip()

        # Drop citations that reference unknown evidence_ids
        valid_ids = _valid_evidence_ids(state.get("final_evidence"))
        filtered = []
        for c in citations:
            if isinstance(c, dict) and c.get("evidence_id") in valid_ids:
                filtered.append(c)

        meta = state.get("answer_meta") or {}
        used_ids = meta.get("used_evidence_ids") or []
        used_ids = [x for x in used_ids if x in valid_ids]

        meta["used_evidence_ids"] = used_ids

        return {
            "final_answer": answer if answer else "(No answer produced.)",
            "citations": filtered,
            "answer_meta": meta,
        }

    return postprocess_answer
