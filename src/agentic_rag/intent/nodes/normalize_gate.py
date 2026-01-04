# src/agentic_rag/intent/nodes/normalize_gate.py

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError

from agentic_rag.intent.prompts.normalize import NORMALIZE_PROMPT
from agentic_rag.intent.state import Clarification, Constraints, Guardrails, IntakeState

logger = logging.getLogger(__name__)

OBSERVE_ENABLED = os.getenv("LANGFUSE_ENABLED", "1") == "1"

if OBSERVE_ENABLED:
    from langfuse.decorators import observe
else:

    def observe(fn=None, **kwargs):
        def _wrap(f):
            return f

        return _wrap(fn) if fn else _wrap


class NormalizeModel(BaseModel):
    normalized_query: str = Field(..., min_length=1)
    constraints: Constraints = Field(default_factory=dict)
    guardrails: Guardrails = Field(default_factory=dict)
    clarification: Clarification = Field(default_factory=dict)
    language: Optional[str] = None
    locale: Optional[str] = None


def make_normalize_gate_node(llm):
    # Build prompt once (faster, less error-prone)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NORMALIZE_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )

    # Bind structured output to the node-specific schema
    model = llm.with_structured_output(NormalizeModel)

    # Compose once
    chain = prompt | model

    @observe
    def intake_normalize(state: IntakeState) -> Dict[str, Any]:
        user_messages = state.get("messages")

        if not isinstance(user_messages, list) or len(user_messages) == 0:
            return {
                "errors": [
                    {
                        "node": "normalize_gate",
                        "type": "schema_validation",
                        "message": "Missing or invalid 'messages' in state (expected non-empty list).",
                        "retryable": False,
                        "details": {"messages_type": str(type(user_messages))},
                    }
                ]
            }

        try:
            # Use direct invocation instead of | pipe for better testability and stability with mocks
            prompt_val = prompt.invoke({"messages": user_messages})
            raw = model.invoke(prompt_val)

            # Support both dict and Pydantic object
            if isinstance(raw, NormalizeModel):
                result = raw
            elif hasattr(raw, "model_dump"):  # Handle potential duck-typing/other models
                result = NormalizeModel.model_validate(raw.model_dump())
            else:
                result = NormalizeModel.model_validate(raw)
        except ValidationError as e:
            # model output didn't match schema
            return {
                "errors": [
                    {
                        "node": "normalize_gate",
                        "type": "model_output_parse",
                        "message": "Structured output failed validation.",
                        "retryable": True,  # often worth retrying once with same prompt
                        "details": {"validation_errors": e.errors()},
                    }
                ]
            }
        except Exception as e:
            # transport/runtime errors
            return {
                "errors": [
                    {
                        "node": "normalize_gate",
                        "type": "runtime_error",
                        "message": str(e),
                        "retryable": True,
                        "details": None,
                    }
                ]
            }

        # Return only fields owned by Node 1
        out: Dict[str, Any] = {
            "normalized_query": result.normalized_query,
            "constraints": result.constraints,
            "guardrails": result.guardrails,
            "clarification": result.clarification,
            "intake_version": "intake_v1",  # Track version for evaluation stability
        }

        # Include only if present (keeps state tidy)
        if result.language is not None:
            out["language"] = result.language
        if result.locale is not None:
            out["locale"] = result.locale

        return out

    return intake_normalize
