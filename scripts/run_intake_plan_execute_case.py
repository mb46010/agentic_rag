# scripts/run_intake_plan_execute_case.py

import argparse
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.json_utils import write_artifact
from scripts.case_utils import resolve_cases, get_case_id

from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.executor.graph import make_executor_graph
from agentic_rag.executor.adapters import (
    NotImplementedRetriever,
    NotImplementedHyDE,
    NotImplementedReranker,
    SimpleRRF,
    NoOpCoverageGrader,
)
from agentic_rag.model import get_default_model


def run_single_case(
    case_path: Path,
    case: Dict[str, Any],
    *,
    run_id: str,
    max_retries: int,
    skip_executor: bool,
    llm,
) -> None:
    """Run the full pipeline for a single case."""
    case_id = get_case_id(case_path, case)

    # Step 1: Run intake graph
    print(f"\n[1/3] Running intake graph for case: {case_id}")
    intake_graph = make_intake_graph(llm, max_retries=max_retries)

    intake_state_in = {
        "messages": case["messages"],
    }

    intake_out = intake_graph.invoke(intake_state_in)
    print(f"  ✓ Intake completed")

    # Check for intake errors
    if intake_out.get("errors"):
        print(f"  ⚠ Intake produced errors: {intake_out['errors']}")

    # Step 2: Run planner graph with intake output
    print(f"[2/3] Running planner graph for case: {case_id}")
    planner_graph = make_planner_graph(llm, max_retries=max_retries)

    # The intake output becomes the planner input
    # (IntakeState is a superset of what planner needs)
    planner_out = planner_graph.invoke(intake_out)
    print(f"  ✓ Planner completed")

    # Check for planner errors
    if planner_out.get("errors"):
        print(f"  ⚠ Planner produced errors: {planner_out['errors']}")

    # Step 3: Run executor graph (if not skipped)
    executor_out = None
    if not skip_executor:
        print(f"[3/3] Running executor graph for case: {case_id}")

        # Create executor graph with placeholder adapters
        # NOTE: If you have real retrieval infrastructure, replace these with actual adapters
        executor_graph = make_executor_graph(
            retriever=NotImplementedRetriever(),
            fusion=SimpleRRF(),
            reranker=NotImplementedReranker(),
            hyde=NotImplementedHyDE(),
            grader=NoOpCoverageGrader(),
            max_retries=max_retries,
        )

        try:
            # The planner output becomes the executor input
            executor_out = executor_graph.invoke(planner_out)
            print(f"  ✓ Executor completed")

            # Check for executor errors
            if executor_out.get("errors"):
                print(f"  ⚠ Executor produced errors: {executor_out['errors']}")
        except NotImplementedError as e:
            print(f"  ⚠ Executor skipped: {e}")
            print(f"    (This is expected if retrieval adapters are not configured)")
            executor_out = {"error": str(e), "message": "Executor requires retrieval adapters"}
    else:
        print(f"[3/3] Executor skipped (--skip-executor flag)")

    # Write artifacts
    artifacts_dir = Path("artifacts/full_pipeline_eval")
    write_artifact(artifacts_dir, run_id, case_id, "input", case)
    write_artifact(artifacts_dir, run_id, case_id, "intake_output", intake_out)
    write_artifact(artifacts_dir, run_id, case_id, "planner_output", planner_out)

    if executor_out is not None:
        write_artifact(artifacts_dir, run_id, case_id, "executor_output", executor_out)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pipeline Summary for: {case_id}")
    print(f"{'='*60}")

    # Intake summary
    if "user_intent" in intake_out:
        print(f"\n[Intake]")
        print(f"  User intent: {intake_out.get('user_intent')}")
        print(f"  Retrieval intent: {intake_out.get('retrieval_intent', 'N/A')}")
        print(f"  Answerability: {intake_out.get('answerability', 'N/A')}")
        print(f"  Normalized query: {intake_out.get('normalized_query', 'N/A')}")

    # Planner summary
    if "plan" in planner_out:
        plan = planner_out["plan"]
        print(f"\n[Planner]")
        print(f"  Strategy: {plan.get('strategy')}")

        if plan.get("clarifying_questions"):
            print(f"  Clarifying questions: {len(plan['clarifying_questions'])}")

        if plan.get("retrieval_rounds"):
            print(f"  Retrieval rounds: {len(plan['retrieval_rounds'])}")
            for i, round_info in enumerate(plan["retrieval_rounds"]):
                purpose = round_info.get('purpose', 'N/A')
                variants = len(round_info.get('query_variants', []))
                print(f"    Round {i}: {purpose} - {variants} variants")

        if plan.get("literal_constraints", {}).get("must_preserve_terms"):
            print(f"  Literal terms: {plan['literal_constraints']['must_preserve_terms']}")

    # Executor summary
    if executor_out and "evidence_pack" in executor_out:
        pack = executor_out["evidence_pack"]
        print(f"\n[Executor]")
        print(f"  Selected evidence: {len(pack.get('selected_evidence', []))} documents")
        print(f"  Rounds completed: {len(executor_out.get('rounds', []))}")

    print(f"\n{'='*60}")
    print(f"Artifacts written to: {artifacts_dir / run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline (intake → planner → executor) on case(s) and persist outputs.\n\n"
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
    parser.add_argument(
        "--skip-executor",
        action="store_true",
        help="Skip executor step (useful when retrieval adapters are not configured)",
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
                skip_executor=args.skip_executor,
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
