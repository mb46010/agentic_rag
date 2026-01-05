from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from agentic_rag.intent.state import IntakeState
from agentic_rag.planner.prompts.planner import PLANNER_PROMPT
from agentic_rag.planner.state import PlannerState

logger = logging.getLogger(__name__)


def make_planner_node(llm):
    """Planner node:
    - Reads IntakeState fields
    - Emits `plan` as a dict (validated against PlannerState)
    - Does not do retrieval or answering
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            # Keep conversation context available, but the schema inputs below are what we actually pass.
            MessagesPlaceholder("messages"),
        ]
    )

    model = llm.with_structured_output(PlannerState, method="function_calling")
    chain = prompt | model

    def planner(state: IntakeState) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return {
                "errors": [
                    {
                        "node": "planner",
                        "type": "schema_validation",
                        "message": "Missing or invalid 'messages' in state (expected non-empty list).",
                        "retryable": False,
                        "details": {"messages_type": str(type(msgs))},
                    }
                ]
            }

        payload = {
            "messages": msgs,
            "normalized_query": state.get("normalized_query", ""),
            "constraints": state.get("constraints") or {},
            "guardrails": state.get("guardrails") or {},
            "clarification": state.get("clarification") or {},
            "user_intent": state.get("user_intent", None),
            "retrieval_intent": state.get("retrieval_intent", None),
            "answerability": state.get("answerability", None),
            "complexity_flags": state.get("complexity_flags") or [],
            "signals": state.get("signals") or {},
            "language": state.get("language", None),
            "locale": state.get("locale", None),
        }

        try:
            raw = chain.invoke(payload)
            plan_obj = PlannerState.model_validate(raw)
        except ValidationError as e:
            return {
                "errors": [
                    {
                        "node": "planner",
                        "type": "model_output_parse",
                        "message": "Planner structured output failed validation.",
                        "retryable": True,
                        "details": {"validation_errors": e.errors()},
                    }
                ]
            }
        except Exception as e:
            return {
                "errors": [
                    {
                        "node": "planner",
                        "type": "runtime_error",
                        "message": str(e),
                        "retryable": True,
                        "details": None,
                    }
                ]
            }

        # Enforce a couple of invariants defensively (belt and suspenders)
        if any(q.blocking for q in plan_obj.clarifying_questions):
            if plan_obj.strategy != "clarify_then_retrieve":
                logger.warning("Forcing strategy=clarify_then_retrieve due to blocking clarifications.")
                plan_obj.strategy = "clarify_then_retrieve"
                plan_obj.retrieval_rounds = []

        if plan_obj.strategy != "retrieve_then_answer":
            plan_obj.retrieval_rounds = []

        return {"plan": plan_obj.model_dump()}

    return planner
