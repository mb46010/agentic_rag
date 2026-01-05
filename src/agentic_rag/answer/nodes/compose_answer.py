# src/agentic_rag/answer/nodes/compose_answer.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from agentic_rag.answer.prompts.compose_answer import COMPOSE_ANSWER_PROMPT
from agentic_rag.answer.state import AnswerState, ComposeAnswerModel, CoverageModel, EvidenceItem

# Optional langfuse decorator - safe when disabled
try:
    from langfuse import observe  # type: ignore
except Exception:  # pragma: no cover

    def observe(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


logger = logging.getLogger(__name__)


def _coerce_evidence(raw: Any) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    if not isinstance(raw, list):
        return items
    for x in raw:
        try:
            items.append(EvidenceItem.model_validate(x))
        except Exception:
            continue
    return items


def _coerce_coverage(raw: Any) -> CoverageModel:
    try:
        return CoverageModel.model_validate(raw or {})
    except Exception:
        return CoverageModel(confidence=0.0, covered=[], missing=[], contradictions=[])


def make_compose_answer_node(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPOSE_ANSWER_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )

    model = llm.with_structured_output(ComposeAnswerModel, method="function_calling")
    chain = prompt | model

    @observe
    def compose_answer(state: AnswerState) -> Dict[str, Any]:
        mode = state.get("answer_mode", "answer")
        plan = state.get("plan") or {}
        constraints = state.get("constraints") or {}
        guardrails = state.get("guardrails") or {}

        msgs = state.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return {
                "errors": [
                    {
                        "node": "compose_answer",
                        "type": "schema_validation",
                        "message": "Missing or invalid 'messages' in state (expected non-empty list).",
                        "retryable": False,
                        "details": {"messages_type": str(type(msgs))},
                    }
                ]
            }

        evidence = _coerce_evidence(state.get("final_evidence"))
        coverage = _coerce_coverage(state.get("coverage"))

        # If gate decided clarify/refuse, we still use the composer to produce the user-facing text
        # but provide mode explicitly so it doesn't try to answer.
        payload = {
            "messages": msgs,
            "answer_mode": mode,
            "plan": plan,
            "constraints": constraints,
            "guardrails": guardrails,
            "normalized_query": state.get("normalized_query", ""),
            "final_evidence": [e.model_dump() for e in evidence],
            "coverage": coverage.model_dump(),
            "language": state.get("language", None),
            "locale": state.get("locale", None),
        }

        try:
            raw = chain.invoke(payload)
            out = ComposeAnswerModel.model_validate(raw)
        except ValidationError as e:
            return {
                "errors": [
                    {
                        "node": "compose_answer",
                        "type": "model_output_parse",
                        "message": "Answer structured output failed validation.",
                        "retryable": True,
                        "details": {"validation_errors": e.errors()},
                    }
                ]
            }
        except Exception as e:
            return {
                "errors": [
                    {
                        "node": "compose_answer",
                        "type": "runtime_error",
                        "message": str(e),
                        "retryable": True,
                        "details": None,
                    }
                ]
            }

        return {
            "final_answer": out.final_answer,
            "citations": [c.model_dump() for c in out.citations],
            "followups": out.followups,
            "answer_meta": {
                **(state.get("answer_meta") or {}),
                "used_evidence_ids": out.used_evidence_ids,
                "asked_clarification": bool(out.asked_clarification),
                "refusal": bool(out.refusal),
            },
        }

    return compose_answer
