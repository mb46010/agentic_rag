# scripts/run_plan_case.py

import argparse
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.json_utils import write_artifact
from scripts.case_utils import resolve_cases, get_case_id

from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.model import get_default_model


def run_single_case(
    case_path: Path,
    case: Dict[str, Any],
    *,
    run_id: str,
    max_retries: int,
    llm,
) -> None:
    """Run planner graph for a single case."""
    case_id = get_case_id(case_path, case)

    graph = make_planner_graph(llm, max_retries=max_retries)

    # Build state_in from case (should have messages and intake outputs)
    state_in = {
        "messages": case["messages"],
        "normalized_query": case.get("normalized_query", ""),
        "constraints": case.get("constraints", {}),
        "guardrails": case.get("guardrails", {}),
        "clarification": case.get("clarification", {}),
        "user_intent": case.get("user_intent", ""),
        "retrieval_intent": case.get("retrieval_intent", ""),
        "answerability": case.get("answerability", ""),
        "complexity_flags": case.get("complexity_flags", []),
        "signals": case.get("signals", {}),
        "language": case.get("language"),
        "locale": case.get("locale"),
    }

    print(f"\nRunning planner graph for case: {case_id}")
    out = graph.invoke(state_in)
    print(f"  ✓ Completed")

    # Check for errors
    if out.get("errors"):
        print(f"  ⚠ Produced errors: {out['errors']}")

    artifacts_dir = Path("artifacts/planner_eval")
    write_artifact(artifacts_dir, run_id, case_id, "input", case)
    write_artifact(artifacts_dir, run_id, case_id, "final_state", out)

    # Also write just the plan for easier inspection
    if "plan" in out:
        write_artifact(artifacts_dir, run_id, case_id, "plan", out["plan"])
        print(f"  Plan strategy: {out['plan'].get('strategy')}")

    print(f"Artifacts written to: {artifacts_dir / run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")


def main():
    parser = argparse.ArgumentParser(
        description="Run planner graph on case(s) and persist outputs.\n\n"
        "Supports both single cases and glob patterns:\n"
        "  --case tests/planner_eval/cases/planner_v1/c001_*.json\n"
        "  --case 'tests/**/cases/**/*.json'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Path to case JSON or glob pattern (e.g., tests/planner_eval/cases/planner_v1/c001_*.json)",
    )
    parser.add_argument(
        "--run-id",
        default="manual",
        help="Run id for artifacts (default: manual)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Max retries for graph execution (default: 1, no retries)",
    )

    args = parser.parse_args()

    # Resolve cases (supports both single file and glob patterns)
    cases = resolve_cases(args.case)
    print(f"Found {len(cases)} case(s) to process")

    llm = get_default_model()

    # Process each case
    for i, (case_path, case_data) in enumerate(cases, 1):
        if len(cases) > 1:
            print(f"\n{'='*60}")
            print(f"Processing case {i}/{len(cases)}: {case_path.name}")
            print(f"{'='*60}")

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
            if len(cases) == 1:
                raise
            # Continue with other cases
            continue

    print(f"\n{'='*60}")
    print(f"✓ Completed {len(cases)} case(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
