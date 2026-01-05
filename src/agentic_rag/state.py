# src/agentic_rag/state.py
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Import specific state definitions from sub-modules
# We use Any/Dict for flexibility where strict typing causes circular imports or rigid coupling,
# but ideally we align with the keys used in subgraphs.


# Reducer to append errors across nodes (same as in intake/state.py)
def add_errors(existing: Optional[List[Dict[str, Any]]], new: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not existing:
        existing = []
    if not new:
        return existing
    return existing + new


class AgentState(TypedDict, total=False):
    """Unified state for the Master Graph.

    This state is a superset of IntakeState, PlannerState (nested), ExecutorState, and AnswerState.
    """

    # --- INTAKE ---
    # Raw inputs
    messages: Annotated[list, add_messages]
    user_email: str
    user_context_info: Optional[Dict[str, Any]]
    conversation_summary: Optional[str]

    # Intake outputs
    normalized_query: str
    constraints: Dict[str, Any]
    guardrails: Dict[str, Any]
    clarification: Dict[str, Any]
    language: Optional[str]
    locale: Optional[str]

    # Signals
    user_intent: str
    retrieval_intent: str
    answerability: str
    complexity_flags: List[str]
    signals: Dict[str, Any]
    intake_version: str
    debug_notes: Optional[str]

    # --- PLANNER ---
    # The planner subgraph returns a 'plan' dictionary (PlannerState).
    # We store it here to pass to Executor and Answer.
    plan: Optional[Dict[str, Any]]  # keys: goal, strategy, retrieval_rounds, etc.

    # --- EXECUTOR ---
    # Executor keys needed for inter-node communication within creating the plan
    # (though typically Executor is a self-contained subgraph, the main graph needs to hold its outputs).

    # Executor outputs stored in the main state for the Answer stage:
    final_evidence: List[Dict[str, Any]]
    coverage: Dict[str, Any]
    retrieval_report: Dict[str, Any]

    # Executor internal loop state (if we were flattening the graph, but for subgraph we might just need outputs)
    # However, since we pass the *entire* state to subgraphs, we include keys that might populate back up.

    # --- ANSWER ---
    answer_mode: str  # "answer" | "clarify" | "refuse"
    final_answer: str
    citations: List[Dict[str, Any]]
    followups: List[str]
    answer_meta: Dict[str, Any]

    # --- COMMON ---
    # Shared error channel
    errors: Annotated[List[Dict[str, Any]], add_errors]
