# src/agentic_rag/intent/state.py

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


# reducer to append errors across nodes
def add_errors(existing: Optional[List[Dict[str, Any]]], new: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not existing:
        existing = []
    if not new:
        return existing
    return existing + new


# -------------------------
# Small reusable primitives
# -------------------------

TimeSensitivity = Literal["none", "low", "high"]
ContextDependency = Literal["none", "weak", "strong"]
Sensitivity = Literal["normal", "elevated", "restricted"]
Answerability = Literal["internal_corpus", "external", "user_context", "reasoning_only", "mixed"]

UserIntent = Literal[
    "explain",  # explain/teach/overview
    "lookup",  # direct factual lookup
    "compare",  # compare options/approaches
    "decide",  # recommend/choose with criteria
    "troubleshoot",  # diagnose issue
    "summarize",  # summarize provided content
    "extract",  # extract fields/structure from content
    "draft",  # write doc/email/spec based on info
    "plan",  # plan project/steps
    "other",
]

RetrievalIntent = Literal[
    "none",  # no retrieval expected
    "definition",  # definitions / concepts
    "procedure",  # how-to / steps
    "evidence",  # collect citations to support claims
    "examples",  # code/examples/patterns
    "verification",  # verify a claim vs sources
    "background",  # broad context
    "mixed",
]

ArtifactFlag = Literal[
    "has_code",
    "has_stacktrace",
    "has_ids",
    "has_paths",
    "has_urls",
    "has_table",
    "has_quoted_strings",
]

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

ComplexityFlag = Literal[
    "multi_intent",
    "multi_domain",
    "requires_synthesis",
    "requires_strict_precision",
    "long_query",
]


# -------------------------
# Constraints vocabularies
# -------------------------

ConstraintDomain = Literal[
    # keep this short - only domains that matter for planning/routing
    "azure",
    "kubernetes",
    "langgraph",
    "langchain",
    "llamaindex",
    "rag",
    "vector_db",
    "search",
    "compliance",
    "security",
    "other",
]

ConstraintFormat = Literal[
    "no_code",
    "code_ok",
    "bullet_points",
    "step_by_step",
    "table_ok",
    "json_only",
    "short",
    "long",
]

ConstraintProhibition = Literal[
    "no_web_browse",
    "no_external_tools",
    "no_persistent_storage",
    "no_pii",
]

ConstraintNonFunctional = Literal[
    "low_latency",
    "low_cost",
    "privacy_high",
    "deterministic",
    "high_recall",
    "high_precision",
]

# -------------------------
# Structured extractions
# -------------------------


class Constraints(TypedDict, total=False):
    # Things that should constrain future planning/execution.
    domain: List[ConstraintDomain]
    format: List[ConstraintFormat]
    prohibitions: List[ConstraintProhibition]
    nonfunctional: List[ConstraintNonFunctional]


class Entity(TypedDict, total=False):
    # Keep this lightweight; planner can do deeper typing later.
    text: str
    type: Literal["product", "component", "org", "person", "doc_type", "concept", "other"]
    confidence: Literal["low", "medium", "high"]


class Acronym(TypedDict, total=False):
    text: str
    expansion: Optional[str]  # None if unknown
    confidence: Literal["low", "medium", "high"]


class Guardrails(TypedDict, total=False):
    time_sensitivity: TimeSensitivity
    context_dependency: ContextDependency
    sensitivity: Sensitivity
    pii_present: bool


class Signals(TypedDict, total=False):
    entities: List[Entity]
    acronyms: List[Acronym]
    artifact_flags: List[ArtifactFlag]
    literal_terms: List[str]  # exact strings to preserve (ids, errors, quoted terms)


class Clarification(TypedDict, total=False):
    needed: bool
    blocking: bool
    reasons: List[ClarificationReason]


class IntakeError(TypedDict):
    node: str  # eg "normalize_gate"
    type: str  # eg "schema_validation", "model_output_parse"
    message: str
    retryable: bool
    details: Optional[Dict[str, Any]]  # optional extra context (keep small)


# -------------------------
# Intake subgraph state
# -------------------------


class IntakeState(TypedDict, total=False):
    # Raw inputs
    messages: Annotated[list, add_messages]
    user_email: str
    user_context_info: Optional[Dict[str, Any]]  # optional from app (could be graph API, job title, ACL)
    # user_message: str
    conversation_summary: Optional[str]  # or a pointer/id, depending on your architecture

    # Node 1: Normalize + Gate
    normalized_query: str
    # constraints: Constraints
    # guardrails: Guardrails
    # clarification: Clarification
    constraints: Dict[str, Any]  # or Constraints if you import it here
    guardrails: Dict[str, Any]  # or Guardrails
    clarification: Dict[str, Any]  # or Clarification
    language: Optional[str]
    locale: Optional[str]

    # Node 2: Signals for Planning
    user_intent: str
    retrieval_intent: str
    answerability: str
    complexity_flags: List[str]
    # needed by .with_structured_output
    signals: Signals
    # user_intent: UserIntent
    # retrieval_intent: RetrievalIntent
    # answerability: Answerability
    # complexity_flags: List[ComplexityFlag]
    # signals: Signals

    # Meta for logging/traceability
    intake_version: str  # eg "intake_v1"
    debug_notes: Optional[str]  # avoid putting chain-of-thought here; keep it short

    # Error handling (APPEND semantics across nodes)
    errors: Annotated[List[IntakeError], add_errors]
