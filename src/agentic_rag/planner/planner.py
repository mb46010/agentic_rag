# src/agentic_rag/planner/planner.py
"""Planner node stub.

TODO: Implement planner that converts intake outputs to executable plan.

Expected interface:
    def make_planner_node(llm) -> Callable:
        def planner(state: IntakeState) -> PlannerState:
            # Analyze intake outputs
            # Decide strategy
            # Generate retrieval rounds
            # Set stop conditions
            return plan_dict
        return planner
"""

from typing import Any, Dict


def make_stub_planner_node():
    """Stub planner that generates a simple retrieve-then-answer plan.

    TODO: Replace with LLM-based planner.
    """

    def stub_planner(state: Dict[str, Any]) -> Dict[str, Any]:
        # Very basic stub: always return a simple retrieval plan
        return {
            "plan": {
                "goal": state.get("normalized_query", ""),
                "strategy": "retrieve_then_answer",
                "retrieval_rounds": [
                    {
                        "round_id": 0,
                        "purpose": "recall",
                        "query_variants": [state.get("normalized_query", "")],
                        "retrieval_modes": [{"type": "hybrid", "k": 20, "alpha": 0.5}],
                        "filters": {},
                        "use_hyde": False,
                        "rrf": True,
                        "rerank": {"enabled": True, "model": "cross_encoder", "rerank_top_k": 60},
                        "output": {"max_docs": 8},
                    }
                ],
                "literal_constraints": {"must_preserve_terms": [], "must_match_exactly": False},
                "acceptance_criteria": {
                    "min_independent_sources": 1,
                    "require_authoritative_source": False,
                    "must_cover_entities": [],
                    "must_answer_subquestions": [],
                },
                "stop_conditions": {"max_rounds": 1, "max_total_docs": 12, "no_new_information_rounds": 1},
                "answer_requirements": {"format": [], "tone": "professional", "length": "concise"},
                "budget": {},
                "planner_meta": {"planner_version": "stub_v1", "rationale_tags": []},
            }
        }

    return stub_planner
