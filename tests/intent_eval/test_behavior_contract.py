"""Test behavior contract:

Expected file example:
{
  "constraints_format_contains": ["no_code"],
  "clarification": { "needed": false, "blocking": false },
  "user_intent": "explain",
  "retrieval_intent": "procedure",
  "answerability": "internal_corpus"
}

"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from json_utils import write_artifact

from agentic_rag.intent.graph import make_intake_graph

CASES_DIR = Path("tests/intent_eval/cases/intake_v1")
EXPECTED_DIR = Path("tests/intent_eval/expected/intake_v1")
ARTIFACTS_DIR = Path("artifacts/intent_eval")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_case_files() -> List[Path]:
    if not CASES_DIR.exists():
        return []
    return sorted([p for p in CASES_DIR.glob("*.json") if p.is_file()])


def _expected_path_for(case_path: Path) -> Path:
    return EXPECTED_DIR / f"{case_path.stem}.expected.json"


def _run_intake_graph(llm, case: Dict[str, Any]) -> Dict[str, Any]:
    graph = make_intake_graph(llm, max_retries=3)
    state_in = {"messages": case["messages"]}
    if "conversation_summary" in case:
        state_in["conversation_summary"] = case["conversation_summary"]
    if "user_context_info" in case:
        state_in["user_context_info"] = case["user_context_info"]
    return graph.invoke(state_in)


def _assert_subset(actual_list: List[Any], expected_subset: List[Any], field: str) -> None:
    missing = [x for x in expected_subset if x not in actual_list]
    assert not missing, f"{field}: missing expected items {missing}. actual={actual_list}"


@pytest.mark.parametrize("case_path", _list_case_files())
def test_behavior_contract_hard_fail(case_path: Path, llm):
    """Hard-fail behavior checks for a SMALL set of labeled fields.
    Expected files should only include what you want to assert.
    """
    expected_path = _expected_path_for(case_path)
    if not expected_path.exists():
        pytest.skip(f"No expected file for {case_path.name}")

    case = _load_json(case_path)
    expected = _load_json(expected_path)

    case_id = case.get("case_id", case_path.stem)
    run_id = os.environ.get("INTENT_EVAL_RUN_ID", "local")

    out = _run_intake_graph(llm, case)

    write_artifact(ARTIFACTS_DIR, run_id, case_id, "input", case)
    write_artifact(ARTIFACTS_DIR, run_id, case_id, "expected", expected)
    write_artifact(ARTIFACTS_DIR, run_id, case_id, "final_state", out)

    errors = out.get("errors") or []
    assert errors == [], f"Graph returned errors: {errors}"

    # ---- Checks (only if present in expected) ----

    # normalized_query exact or contains checks
    nq_exp = expected.get("normalized_query")
    if isinstance(nq_exp, str):
        assert out["normalized_query"] == nq_exp

    nq_contains = expected.get("normalized_query_contains")
    if isinstance(nq_contains, list):
        for needle in nq_contains:
            assert needle in out["normalized_query"], f"normalized_query must contain {needle!r}"

    # constraints.format subset
    fmt_exp = expected.get("constraints_format_contains")
    if isinstance(fmt_exp, list):
        actual_fmt = (out.get("constraints") or {}).get("format") or []
        _assert_subset(actual_fmt, fmt_exp, "constraints.format")

    # clarification checks
    clar_exp = expected.get("clarification")
    if isinstance(clar_exp, dict):
        actual_clar = out.get("clarification") or {}
        if "needed" in clar_exp:
            assert actual_clar.get("needed") == clar_exp["needed"]
        if "blocking" in clar_exp:
            assert actual_clar.get("blocking") == clar_exp["blocking"]
        if "reasons_contains" in clar_exp:
            _assert_subset(actual_clar.get("reasons") or [], clar_exp["reasons_contains"], "clarification.reasons")

    # extract_signals core labels
    for k in ("user_intent", "retrieval_intent", "answerability"):
        if k in expected:
            assert out.get(k) == expected[k], f"{k} mismatch: expected={expected[k]!r} actual={out.get(k)!r}"

    # complexity flags subset
    cf_exp = expected.get("complexity_flags_contains")
    if isinstance(cf_exp, list):
        _assert_subset(out.get("complexity_flags") or [], cf_exp, "complexity_flags")

    # artifact flags subset (if you label them)
    af_exp = expected.get("artifact_flags_contains")
    if isinstance(af_exp, list):
        actual_flags = (out.get("signals") or {}).get("artifact_flags") or []
        _assert_subset(actual_flags, af_exp, "signals.artifact_flags")
