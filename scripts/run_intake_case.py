# scripts/run_intake_case.py

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from tests.intent_eval.utils import write_artifact  # reuse your artifact writer

from agentic_rag.intent.graph import make_intake_graph
from agentic_rag.model import get_default_model


def load_case(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run intake graph on a single case and persist output.")
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
    graph = make_intake_graph(llm, max_retries=args.max_retries)

    state_in = {
        "messages": case["messages"],
    }

    print(f"Running intake graph for case: {case_id}")
    out = graph.invoke(state_in)

    artifacts_dir = Path("artifacts/intent_eval")
    write_artifact(artifacts_dir, args.run_id, case_id, "input", case)
    write_artifact(artifacts_dir, args.run_id, case_id, "final_state", out)

    print(f"Artifacts written to: {artifacts_dir / args.run_id / case_id}")
    print("Review output and create/update expected JSON accordingly.")


if __name__ == "__main__":
    main()
