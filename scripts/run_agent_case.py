# scripts/run_agent_case.py

import argparse
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Import real adapters (or mocks depending on environment, but let's assume we want real structure)
# For simplicity in this shell, we might need to instantiate them with dummy or real components.
# If the user environment has these ready, great. If not, we might need simple mocks.
# Let's inspect `agentic_rag.executor.adapters`.
# ... I'll rely on the fact that I can instantiate them if they are simple or mocks.
from agentic_rag.executor.adapters import (
    CoverageGraderAdapter,
    FusionAdapter,
    HyDEAdapter,
    RerankerAdapter,
    RetrieverAdapter,
)
from agentic_rag.graph import make_agent_graph
from agentic_rag.model import get_default_model
from scripts.case_utils import get_case_id, resolve_cases
from tests.intent_eval.json_utils import write_artifact


# Minimal mock implementations for running without full infrastructure if needed,
# but ideally we use what is available.
class MockRetriever(RetrieverAdapter):
    def search(self, query, **kwargs):
        # Return dummy candidate
        return []


class MockFusion(FusionAdapter):
    def merge(self, candidates, **kwargs):
        return candidates


class MockReranker(RerankerAdapter):
    def rerank(self, candidates, **kwargs):
        return candidates


class MockHyDE(HyDEAdapter):
    def generate_hypothetical_docs(self, query, **kwargs):
        return [f"Hypothetical doc for {query}"]


class MockGrader(CoverageGraderAdapter):
    def grade(self, **kwargs):
        return {"confidence": 0.5, "evidence_quality": "medium"}


def run_single_case(
    case_path: Path,
    case: Dict[str, Any],
    *,
    run_id: str,
    max_retries: int,
    llm,
) -> None:
    """Run agent graph for a single case."""
    case_id = get_case_id(case_path, case)

    # Instantiate adapters (using mocks for lightweight verification unless configured otherwise)
    # In a real app, these would be injected via dependency injection or config.
    retriever = MockRetriever()
    fusion = MockFusion()
    reranker = MockReranker()
    hyde = MockHyDE()
    grader = MockGrader()

    graph = make_agent_graph(
        llm,
        max_retries=max_retries,
        retriever=retriever,
        fusion=fusion,
        reranker=reranker,
        hyde=hyde,
        grader=grader,
    )

    state_in = {
        "messages": case["messages"],
        "user_email": "test@example.com",
    }

    print(f"\nRunning AGENT graph for case: {case_id}")
    out = graph.invoke(state_in)

    # Simple output summary
    print("  ✓ Completed")
    if out.get("errors"):
        print(f"  ⚠ Produced errors: {out['errors']}")

    print(f"  ➡ Final Answer: {out.get('final_answer')[:100]}...")

    artifacts_dir = Path("artifacts/agent_eval")
    write_artifact(artifacts_dir, run_id, case_id, "input", case)
    write_artifact(artifacts_dir, run_id, case_id, "final_state", out)

    print(f"Artifacts written to: {artifacts_dir / run_id / case_id}")


def main():
    parser = argparse.ArgumentParser(description="Run MASTER AGENT graph on case(s).")
    parser.add_argument(
        "--case",
        required=True,
        help="Path to case JSON or glob pattern",
    )
    parser.add_argument(
        "--run-id",
        default="manual_agent",
        help="Run id for artifacts",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries",
    )

    args = parser.parse_args()
    cases = resolve_cases(args.case)
    print(f"Found {len(cases)} case(s) to process")

    llm = get_default_model()

    for i, (case_path, case_data) in enumerate(cases, 1):
        try:
            run_single_case(
                case_path,
                case_data,
                run_id=args.run_id,
                max_retries=args.max_retries,
                llm=llm,
            )
        except Exception as e:
            print(f"\n❌ Error processing {case_path.name}: {e}")
            raise


if __name__ == "__main__":
    main()
