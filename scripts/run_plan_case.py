# scripts/run_plan_case.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.json_utils import write_artifact

from agentic_rag.planner.graph import make_planner_graph
from agentic_rag.model import get_default_model


def load_case(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run planner graph on a single case and persist output.")
    parser.add_argument(
        "--case",
        required=True,
        help="Path to planner case JSON (e.g. tests/planner_eval/cases/planner_v1/c001_*.json)",
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
    graph = make_planner_graph(llm, max_retries=args.max_retries)

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

    print(f"Running planner graph for case: {case_id}")
    out = graph.invoke(state_in)

    artifacts_dir = Path("artifacts/planner_eval")
    write_artifact(artifacts_dir, args.run_id, case_id, "input", case)
    write_artifact(artifacts_dir, args.run_id, case_id, "final_state", out)

    # Also write just the plan for easier inspection
    if "plan" in out:
        write_artifact(artifacts_dir, args.run_id, case_id, "plan", out["plan"])

    print(f"Artifacts written to: {artifacts_dir / args.run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")


if __name__ == "__main__":
    main()
