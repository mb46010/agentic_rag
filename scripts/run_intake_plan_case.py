# scripts/run_intake_plan_case.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.json_utils import write_artifact

from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.model import get_default_model


def load_case(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run intake graph followed by planner graph on a single case and persist output."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Path to intake case JSON (e.g. tests/intent_eval/cases/intake_v1/c001_*.json)",
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

    case_path = Path(args.case)
    if not case_path.exists():
        raise FileNotFoundError(case_path)

    case = load_case(case_path)
    case_id = case.get("case_id", case_path.stem)

    llm = get_default_model()

    # Step 1: Run intake graph
    print(f"[1/2] Running intake graph for case: {case_id}")
    intake_graph = make_intake_graph(llm, max_retries=args.max_retries)

    intake_state_in = {
        "messages": case["messages"],
    }

    intake_out = intake_graph.invoke(intake_state_in)
    print(f"  ✓ Intake completed")

    # Check for intake errors
    if intake_out.get("errors"):
        print(f"  ⚠ Intake produced errors: {intake_out['errors']}")

    # Step 2: Run planner graph with intake output
    print(f"[2/2] Running planner graph for case: {case_id}")
    planner_graph = make_planner_graph(llm, max_retries=args.max_retries)

    # The intake output becomes the planner input
    # (IntakeState is a superset of what planner needs)
    planner_out = planner_graph.invoke(intake_out)
    print(f"  ✓ Planner completed")

    # Check for planner errors
    if planner_out.get("errors"):
        print(f"  ⚠ Planner produced errors: {planner_out['errors']}")

    # Write artifacts
    artifacts_dir = Path("artifacts/intake_plan_eval")
    write_artifact(artifacts_dir, args.run_id, case_id, "input", case)
    write_artifact(artifacts_dir, args.run_id, case_id, "intake_output", intake_out)
    write_artifact(artifacts_dir, args.run_id, case_id, "planner_output", planner_out)

    # Also write just the plan for easier inspection
    if "plan" in planner_out:
        write_artifact(artifacts_dir, args.run_id, case_id, "plan", planner_out["plan"])
        print(f"\n✓ Plan strategy: {planner_out['plan'].get('strategy')}")

        # Print key plan details
        plan = planner_out["plan"]
        if plan.get("clarifying_questions"):
            print(f"  Clarifying questions: {len(plan['clarifying_questions'])}")
        if plan.get("retrieval_rounds"):
            print(f"  Retrieval rounds: {len(plan['retrieval_rounds'])}")
            for i, round_info in enumerate(plan["retrieval_rounds"]):
                print(f"    Round {i}: {round_info.get('purpose')} - {len(round_info.get('query_variants', []))} variants")
        if plan.get("literal_constraints", {}).get("must_preserve_terms"):
            print(f"  Literal terms: {plan['literal_constraints']['must_preserve_terms']}")

    print(f"\nArtifacts written to: {artifacts_dir / args.run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")


if __name__ == "__main__":
    main()
