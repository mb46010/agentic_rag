# src/agentic_rag/answer/state.py
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Answer modes
AnswerMode = Literal["answer", "clarify", "refuse"]


# Coverage model
class CoverageModel(BaseModel):
    """Coverage assessment from executor."""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    covered: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)


# Evidence item
class EvidenceItem(BaseModel):
    """Evidence item from executor."""

    evidence_id: str
    text: str = ""
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Citation model
class Citation(BaseModel):
    """Citation for answer."""

    evidence_id: str
    text: str = ""
    source: Optional[str] = None


# Compose answer output model
class ComposeAnswerModel(BaseModel):
    """Output from compose_answer node."""

    final_answer: str
    citations: List[Citation] = Field(default_factory=list)
    followups: List[str] = Field(default_factory=list)
    used_evidence_ids: List[str] = Field(default_factory=list)
    asked_clarification: bool = False
    refusal: bool = False


# Answer state
class AnswerState(TypedDict, total=False):
    """State for answer subgraph."""

    # Inputs from upstream
    messages: List[Any]
    plan: Dict[str, Any]
    guardrails: Dict[str, Any]
    constraints: Dict[str, Any]
    normalized_query: str
    final_evidence: List[Any]
    coverage: Dict[str, Any]
    language: Optional[str]
    locale: Optional[str]

    # Answer gate output
    answer_mode: AnswerMode

    # Compose answer output
    final_answer: str
    citations: List[Dict[str, Any]]
    followups: List[str]

    # Metadata
    answer_meta: Dict[str, Any]

    # Errors
    errors: List[Dict[str, Any]]
