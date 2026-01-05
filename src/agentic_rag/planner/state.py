from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, confloat, conint

# Keep aligned with intent types where possible.
Strategy = Literal[
    "direct_answer",
    "retrieve_then_answer",
    "clarify_then_retrieve",
    "defer_or_refuse",
]

RoundPurpose = Literal["recall", "precision", "verification", "gap_filling"]
RetrievalModeType = Literal["bm25", "vector", "hybrid"]

ClarificationReason = Literal[
    "missing_version",
    "missing_timeframe",
    "missing_scope",
    "ambiguous_acronym",
    "ambiguous_entity",
    "conflicting_constraints",
    "unclear_success_criteria",
    "context_reference_unresolved",
]

Sensitivity = Literal["normal", "elevated", "restricted"]


class ClarifyingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=5)
    reason: ClarificationReason
    blocking: bool = True


class RetrievalModeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RetrievalModeType
    k: conint(ge=1, le=200) = 20
    alpha: Optional[confloat(ge=0.0, le=1.0)] = None


class RoundFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_types: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    time_range: Optional[str] = None


class RerankSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: Literal["cross_encoder"] = "cross_encoder"
    rerank_top_k: conint(ge=5, le=200) = 60


class RoundOutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_docs: conint(ge=1, le=50) = 8


class RetrievalRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_id: conint(ge=0, le=10)
    purpose: RoundPurpose

    query_variants: List[str] = Field(default_factory=list, min_length=1)

    retrieval_modes: List[RetrievalModeSpec] = Field(default_factory=list)

    filters: RoundFilters = Field(default_factory=RoundFilters)

    use_hyde: bool = False
    rrf: bool = True

    rerank: RerankSpec = Field(default_factory=RerankSpec)
    output: RoundOutputSpec = Field(default_factory=RoundOutputSpec)


class LiteralConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must_preserve_terms: List[str] = Field(default_factory=list)
    must_match_exactly: bool = False


class AcceptanceCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_independent_sources: conint(ge=1, le=5) = 1
    require_authoritative_source: bool = False
    must_cover_entities: List[str] = Field(default_factory=list)
    must_answer_subquestions: List[str] = Field(default_factory=list)


class StopConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: conint(ge=1, le=3) = 2
    max_total_docs: conint(ge=1, le=30) = 12
    confidence_threshold: Optional[confloat(ge=0.0, le=1.0)] = None
    no_new_information_rounds: conint(ge=0, le=2) = 1


class AnswerRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: List[str] = Field(default_factory=list)  # e.g. ["bullet_points", "citations", "no_code"]
    tone: Optional[str] = None
    length: Optional[str] = None
    citation_style: Optional[str] = None


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: conint(ge=256, le=100000) = 8000
    max_latency_ms: Optional[conint(ge=100, le=120000)] = None


class SafetySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensitivity: Sensitivity = "normal"
    pii_allowed: bool = False


class PlannerMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner_version: str = "planner_v1"
    rationale_tags: List[str] = Field(default_factory=list)


class PlannerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=5)
    strategy: Strategy

    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)

    retrieval_rounds: List[RetrievalRound] = Field(default_factory=list)
    literal_constraints: LiteralConstraints = Field(default_factory=LiteralConstraints)

    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)

    answer_requirements: AnswerRequirements = Field(default_factory=AnswerRequirements)

    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)

    planner_meta: PlannerMeta = Field(default_factory=PlannerMeta)
