# src/agentic_rag/intent/nodes/extract_signals.py

import logging
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse import observe
from pydantic import BaseModel, ValidationError

from agentic_rag.intent.prompts.extract_signals import EXTRACT_SIGNALS_PROMPT
from agentic_rag.intent.state import IntakeState

logger = logging.getLogger(__name__)


class ExtractSignalsModel(BaseModel):
    user_intent: str
    retrieval_intent: str
    answerability: str
    complexity_flags: List[str]
    signals: Dict[str, Any]


def make_extract_signals_node(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACT_SIGNALS_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )

    model = llm.with_structured_output(ExtractSignalsModel)
    chain = prompt | model

    @observe
    def extract_signals(state: IntakeState) -> Dict[str, Any]:
        user_messages = state.get("messages")
        if not isinstance(user_messages, list) or not user_messages:
            return {
                "errors": [
                    {
                        "node": "extract_signals",
                        "type": "schema_validation",
                        "message": "Missing or invalid 'messages' in state (expected non-empty list).",
                        "retryable": False,
                        "details": {"messages_type": str(type(user_messages))},
                    }
                ]
            }

        # Pull fields from previous normalize_gate step (use conservative defaults)
        normalized_query = state.get("normalized_query", "")
        constraints = state.get("constraints", {}) or {}
        guardrails = state.get("guardrails", {}) or {}
        clarification = state.get("clarification", {}) or {}
        language = state.get("language", None)
        locale = state.get("locale", None)

        try:
            # Depending on LangChain version/model wrapper, this may return a dict OR a BaseModel.
            raw = chain.invoke(
                {
                    "messages": user_messages,
                    "normalized_query": normalized_query,
                    "constraints": constraints,
                    "guardrails": guardrails,
                    "clarification": clarification,
                    "language": language,
                    "locale": locale,
                }
            )

            # Normalize to a Pydantic object for consistent attribute access.
            result = ExtractSignalsModel.model_validate(raw)

        except ValidationError as e:
            return {
                "errors": [
                    {
                        "node": "extract_signals",
                        "type": "model_output_parse",
                        "message": "Structured output failed validation.",
                        "retryable": True,
                        "details": {"validation_errors": e.errors()},
                    }
                ]
            }
        except Exception as e:
            return {
                "errors": [
                    {
                        "node": "extract_signals",
                        "type": "runtime_error",
                        "message": str(e),
                        "retryable": True,
                        "details": None,
                    }
                ]
            }

        return {
            "user_intent": result.user_intent,
            "retrieval_intent": result.retrieval_intent,
            "answerability": result.answerability,
            "complexity_flags": result.complexity_flags,
            "signals": result.signals,
        }

    return extract_signals
