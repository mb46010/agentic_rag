# src/agentic_rag/answer/state.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, confloat, conint

AnswerMode = Literal["answer", "clarify", "refuse"]

CitationStyle = Literal["none", "inline", "footnote"]

# -------------------------
# Evidence + coverage models
# -------------------------


class EvidenceItem(BaseModel):
    """Minimal evidence contract the Answer stage can consume.

    Keep this aligned with what the executor produces, but do not rely on executor internals.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)

    source: Optional[str] = None  # title/uri/display string
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
    scores: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class CoverageModel(BaseModel):
    """Output from executor grading step (or a placeholder/no-op grader initially)."""

    model_config = ConfigDict(extra="forbid")

    confidence: confloat(ge=0.0, le=1.0) = 0.0
    covered: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)


# -------------------------
# Answer output models
# -------------------------


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    text: Optional[str] = None
    source: Optional[str] = None
    # Optional: location within answer text for UI highlighting
    span_start: Optional[conint(ge=0)] = None
    span_end: Optional[conint(ge=0)] = None
    note: Optional[str] = None


class AnswerMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_version: str = "answer_v1"
    mode: AnswerMode = "answer"
    used_evidence_ids: List[str] = Field(default_factory=list)
    coverage_confidence: float = 0.0
    refusal: bool = False
    asked_clarification: bool = False


class ComposeAnswerModel(BaseModel):
    """Structured output returned by the compose_answer LLM step."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    used_evidence_ids: List[str] = Field(default_factory=list)
    followups: List[str] = Field(default_factory=list)
    asked_clarification: bool = False
    refusal: bool = False


# -------------------------
# LangGraph state
# -------------------------


class AnswerState(TypedDict, total=False):
    # From intake
    messages: list
    normalized_query: str
    constraints: Dict[str, Any]
    guardrails: Dict[str, Any]
    language: Optional[str]
    locale: Optional[str]

    # From planner
    plan: Dict[str, Any]  # PlannerState as dict

    # From executor
    final_evidence: List[Dict[str, Any]]  # list of EvidenceItem-like dicts
    coverage: Dict[str, Any]  # CoverageModel-like dict
    retrieval_report: Dict[str, Any]

    # Answer stage outputs
    answer_mode: AnswerMode
    final_answer: str
    citations: List[Dict[str, Any]]
    followups: List[str]
    answer_meta: Dict[str, Any]

    # Shared error channel
    errors: List[Dict[str, Any]]
