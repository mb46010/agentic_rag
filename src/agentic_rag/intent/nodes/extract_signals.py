# src/agentic_rag/intent/nodes/extract_signals.py
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# If this import fails in your env, switch to:
# from langfuse.decorators import observe
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_rag.intent.prompts.extract_signals import EXTRACT_SIGNALS_PROMPT
from agentic_rag.intent.state import (
    Answerability,
    ArtifactFlag,
    ComplexityFlag,
    IntakeState,
    RetrievalIntent,
    UserIntent,
)

logger = logging.getLogger(__name__)

OBSERVE_ENABLED = os.getenv("LANGFUSE_ENABLED", "1") == "1"

if OBSERVE_ENABLED:
    try:
        from langfuse import observe
    except ImportError:

        def observe(fn=None, **kwargs):
            return fn if fn else lambda f: f
else:

    def observe(fn=None, **kwargs):
        def _wrap(f):
            return f

        return _wrap(fn) if fn else _wrap


class EntityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    type: Literal["product", "component", "org", "person", "doc_type", "concept", "other"]
    confidence: Literal["low", "medium", "high"]


class AcronymModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    expansion: Optional[str]  # None if unknown
    confidence: Literal["low", "medium", "high"]


class SignalsModel(BaseModel):
    # For OpenAI structured outputs compatibility and general cleanliness:
    # - default empty lists
    # - forbid extra keys (prevents prompt drift)
    model_config = ConfigDict(extra="forbid")

    entities: List[EntityModel] = Field(default_factory=list)
    acronyms: List[AcronymModel] = Field(default_factory=list)
    artifact_flags: List[ArtifactFlag] = Field(default_factory=list)
    literal_terms: List[str] = Field(default_factory=list)


class ExtractSignalsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_intent: UserIntent
    retrieval_intent: RetrievalIntent
    answerability: Answerability
    complexity_flags: List[ComplexityFlag] = Field(default_factory=list)
    signals: SignalsModel = Field(default_factory=SignalsModel)


def make_extract_signals_node(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACT_SIGNALS_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )

    # Keep this while you iterate; it's tolerant to schema quirks.
    # Once stable, you can try removing method="function_calling" to use strict structured outputs.
    model = llm.with_structured_output(ExtractSignalsModel, method="function_calling")
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

        # Pull fields from normalize_gate step (conservative defaults)
        normalized_query = state.get("normalized_query", "")
        constraints = state.get("constraints") or {}
        guardrails = state.get("guardrails") or {}
        clarification = state.get("clarification") or {}
        language = state.get("language", None)
        locale = state.get("locale", None)

        try:
            # Use direct invocation instead of | pipe for better testability and stability with mocks
            prompt_val = prompt.invoke(
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
            raw = model.invoke(prompt_val)

            # Support both dict and Pydantic object (for testing and LLM variation)
            if isinstance(raw, ExtractSignalsModel):
                result = raw
            elif hasattr(raw, "model_dump"):  # Handle potential duck-typing/other models
                result = ExtractSignalsModel.model_validate(raw.model_dump())
            else:
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

        # Return JSON-serializable values into LangGraph state
        return {
            "user_intent": result.user_intent,
            "retrieval_intent": result.retrieval_intent,
            "answerability": result.answerability,
            "complexity_flags": result.complexity_flags,
            "signals": result.signals.model_dump(),
        }

    return extract_signals
