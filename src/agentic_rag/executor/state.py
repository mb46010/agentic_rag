# src/agentic_rag/executor/state.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

# -------------------------
# Planner contract (input)
# -------------------------

Strategy = Literal[
    "direct_answer",
    "retrieve_then_answer",
    "clarify_then_retrieve",
    "defer_or_refuse",
]

RoundPurpose = Literal["recall", "precision", "verification", "gap_filling"]
RetrievalModeType = Literal["bm25", "vector", "hybrid"]


class RetrievalModeSpec(TypedDict, total=False):
    type: RetrievalModeType
    k: int
    alpha: Optional[float]


class RoundFilters(TypedDict, total=False):
    doc_types: List[str]
    domains: List[str]
    entities: List[str]
    time_range: str  # free-form for now


class RerankSpec(TypedDict, total=False):
    enabled: bool
    model: Literal["cross_encoder"]
    rerank_top_k: int


class RoundOutputSpec(TypedDict, total=False):
    max_docs: int


class RetrievalRound(TypedDict, total=False):
    round_id: int
    purpose: RoundPurpose
    query_variants: List[str]
    retrieval_modes: List[RetrievalModeSpec]
    filters: RoundFilters
    use_hyde: bool
    rrf: bool
    rerank: RerankSpec
    output: RoundOutputSpec


class LiteralConstraints(TypedDict, total=False):
    must_preserve_terms: List[str]
    must_match_exactly: bool


class AcceptanceCriteria(TypedDict, total=False):
    min_independent_sources: int
    require_authoritative_source: bool
    must_cover_entities: List[str]
    must_answer_subquestions: List[str]


class StopConditions(TypedDict, total=False):
    max_rounds: int
    max_total_docs: int
    confidence_threshold: float
    no_new_information_rounds: int


class AnswerRequirements(TypedDict, total=False):
    format: List[str]
    tone: str
    length: str
    citation_style: str


class BudgetSpec(TypedDict, total=False):
    max_tokens: int
    max_latency_ms: int


class PlannerMeta(TypedDict, total=False):
    planner_version: str
    rationale_tags: List[str]


class PlannerState(TypedDict, total=False):
    goal: str
    strategy: Strategy
    clarifying_questions: List[Dict[str, Any]]

    retrieval_rounds: List[RetrievalRound]
    literal_constraints: LiteralConstraints
    acceptance_criteria: AcceptanceCriteria
    stop_conditions: StopConditions
    answer_requirements: AnswerRequirements
    budget: BudgetSpec
    planner_meta: PlannerMeta


# -------------------------
# Evidence / candidates
# -------------------------


@dataclass(frozen=True)
class CandidateKey:
    doc_id: str
    chunk_id: str


@dataclass
class Candidate:
    key: CandidateKey
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Retrieval features
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None

    # Fusion + rerank
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Provenance
    round_id: Optional[int] = None
    query: Optional[str] = None
    mode: Optional[str] = None  # "bm25" | "vector" | "hybrid"


@dataclass
class RoundResult:
    round_id: int
    purpose: str
    queries: List[str] = field(default_factory=list)

    raw_candidates_count: int = 0
    merged_candidates_count: int = 0
    reranked_candidates_count: int = 0

    selected: List[Candidate] = field(default_factory=list)
    novelty_new_items: int = 0

    debug: Dict[str, Any] = field(default_factory=dict)


class Coverage(TypedDict, total=False):
    covered_entities: List[str]
    missing_entities: List[str]
    covered_subquestions: List[str]
    missing_subquestions: List[str]
    evidence_quality: Literal["high", "medium", "low"]
    confidence: float
    contradictions: List[str]


# -------------------------
# Executor state (graph)
# -------------------------


class ExecutorState(TypedDict, total=False):
    # Inputs
    plan: PlannerState
    normalized_query: str  # from intake
    constraints: Dict[str, Any]  # from intake
    guardrails: Dict[str, Any]  # from intake
    signals: Dict[str, Any]  # from intake

    # Execution context
    execution_context: Dict[str, Any]

    # Round loop
    current_round_index: int
    round_queries: List[str]
    round_candidates_raw: List[Candidate]
    round_candidates_merged: List[Candidate]
    round_candidates_reranked: List[Candidate]
    round_selected: List[Candidate]

    # Aggregation
    rounds: List[RoundResult]
    evidence_pool: List[Candidate]
    final_evidence: List[Candidate]
    coverage: Coverage
    retrieval_report: Dict[str, Any]

    # Control flow
    continue_search: bool

    # Errors
    errors: List[Dict[str, Any]]
