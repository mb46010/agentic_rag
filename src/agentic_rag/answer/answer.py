# src/agentic_rag/answer/answer.py
"""Answer synthesis node stub.

TODO: Implement answer synthesis from evidence.

Expected interface:
    def make_answer_node(llm) -> Callable:
        def answer(state: ExecutorState) -> Dict[str, Any]:
            # Take final_evidence
            # Apply answer_requirements
            # Synthesize response with citations
            # Check coverage for blocking issues
            return {"final_response": text}
        return answer
"""

from typing import Any, Dict


def make_stub_answer_node():
    """Stub answer node that formats evidence as a simple response.

    TODO: Replace with LLM-based answer synthesis.
    """

    def stub_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        final_evidence = state.get("final_evidence", [])
        normalized_query = state.get("normalized_query", "")

        if not final_evidence:
            response = f"I couldn't find sufficient evidence to answer: {normalized_query}"
        else:
            # Very basic stub: just concatenate evidence
            evidence_texts = [f"[{i+1}] {c.text[:200]}..." for i, c in enumerate(final_evidence[:3])]
            response = f"Based on the evidence:\n\n" + "\n\n".join(evidence_texts)

        return {"final_response": response}

    return stub_answer
