import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from dotenv import load_dotenv
from json_utils import write_artifact

from agentic_rag.intent.graph import make_intake_graph

load_dotenv()

CASES_DIR = Path("tests/intent_eval/cases/intake_v1")
ARTIFACTS_DIR = Path("artifacts/intent_eval")

# Set via env when you want to tighten later
STABILITY_RUNS = int(os.environ.get("INTENT_EVAL_STABILITY_RUNS", "3"))
FAIL_ON_INSTABILITY = os.environ.get("INTENT_EVAL_FAIL_ON_INSTABILITY", "0") == "1"

MAX_RETRIES = 1


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_case_files() -> List[Path]:
    if not CASES_DIR.exists():
        return []
    return sorted([p for p in CASES_DIR.glob("*.json") if p.is_file()])


def _run_once(llm, case: Dict[str, Any]) -> Dict[str, Any]:
    graph = make_intake_graph(llm, max_retries=MAX_RETRIES)
    state_in = {"messages": case["messages"]}
    if "conversation_summary" in case:
        state_in["conversation_summary"] = case["conversation_summary"]
    if "user_context_info" in case:
        state_in["user_context_info"] = case["user_context_info"]
    return graph.invoke(state_in)


def _stable_projection(out: Dict[str, Any]) -> Dict[str, Any]:
    """Only compare stability-critical fields."""
    clar = out.get("clarification") or {}
    return {
        "user_intent": out.get("user_intent"),
        "retrieval_intent": out.get("retrieval_intent"),
        "answerability": out.get("answerability"),
        "clarification": {
            "needed": clar.get("needed"),
            "blocking": clar.get("blocking"),
            "reasons": sorted((clar.get("reasons") or [])),
        },
        # optional: add constraints.format stability if you want
        "constraints_format": sorted(((out.get("constraints") or {}).get("format") or [])),
    }


def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    diffs = []
    for k in sorted(set(a.keys()) | set(b.keys())):
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)!r} != {b.get(k)!r}")
    return diffs


@pytest.mark.parametrize("case_path", _list_case_files())
def test_stability_soft_fail(case_path: Path, llm):
    case = _load_json(case_path)
    case_id = case.get("case_id", case_path.stem)
    run_id = os.environ.get("INTENT_EVAL_RUN_ID", "local")

    outputs: List[Dict[str, Any]] = []
    projections: List[Dict[str, Any]] = []

    for i in range(STABILITY_RUNS):
        out = _run_once(llm, case)
        outputs.append(out)
        projections.append(_stable_projection(out))

    write_artifact(ARTIFACTS_DIR, run_id, case_id, "stability.input", case)
    write_artifact(ARTIFACTS_DIR, run_id, case_id, "stability.outputs", outputs)
    write_artifact(ARTIFACTS_DIR, run_id, case_id, "stability.projections", projections)

    # Hard fail if any run produced errors (stability is meaningless if it errored)
    for out in outputs:
        errors = out.get("errors") or []
        assert errors == [], f"Graph returned errors: {errors}"

    baseline = projections[0]
    unstable: List[Tuple[int, List[str]]] = []
    for idx, proj in enumerate(projections[1:], start=1):
        d = _diff(baseline, proj)
        if d:
            unstable.append((idx, d))

    if unstable:
        msg = f"Instability detected for {case_id}: " + "; ".join([f"run{idx}({', '.join(d)})" for idx, d in unstable])
        if FAIL_ON_INSTABILITY:
            pytest.fail(msg)
        else:
            pytest.xfail(msg)
