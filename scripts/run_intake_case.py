# scripts/run_intake_case.py

import argparse
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.json_utils import write_artifact
from scripts.case_utils import resolve_cases, get_case_id

from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.model import get_default_model


def run_single_case(
    case_path: Path,
    case: Dict[str, Any],
    *,
    run_id: str,
    max_retries: int,
    llm,
) -> None:
    """Run intake graph for a single case."""
    case_id = get_case_id(case_path, case)

    graph = make_intake_graph(llm, max_retries=max_retries)

    state_in = {
        "messages": case["messages"],
    }

    print(f"\nRunning intake graph for case: {case_id}")
    out = graph.invoke(state_in)
    print(f"  ✓ Completed")

    # Check for errors
    if out.get("errors"):
        print(f"  ⚠ Produced errors: {out['errors']}")

    artifacts_dir = Path("artifacts/intent_eval")
    write_artifact(artifacts_dir, run_id, case_id, "input", case)
    write_artifact(artifacts_dir, run_id, case_id, "final_state", out)

    print(f"Artifacts written to: {artifacts_dir / run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")


def main():
    parser = argparse.ArgumentParser(
        description="Run intake graph on case(s) and persist outputs.\n\n"
        "Supports both single cases and glob patterns:\n"
        "  --case tests/intent_eval/cases/intake_v1/c001_*.json\n"
        "  --case 'tests/**/cases/**/*.json'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Path to case JSON or glob pattern (e.g., tests/intent_eval/cases/intake_v1/c001_*.json)",
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
