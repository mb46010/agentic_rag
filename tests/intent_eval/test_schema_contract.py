import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv
from json_utils import write_artifact

from agentic_rag.intent.graph import make_intake_graph

# ---- Config ----
CASES_DIR = Path("tests/intent_eval/cases/intake_v1")
ARTIFACTS_DIR = Path("artifacts/intent_eval")

# Load .env once for the whole test session
# if os.getenv("PYTEST_CURRENT_TEST"):
load_dotenv()

MAX_RETRIES = 1


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_case_files() -> List[Path]:
    if not CASES_DIR.exists():
        return []
    return sorted([p for p in CASES_DIR.glob("*.json") if p.is_file()])


def _run_intake_graph(llm, case: Dict[str, Any]) -> Dict[str, Any]:
    graph = make_intake_graph(llm, max_retries=MAX_RETRIES)
    # IntakeState expects "messages" at minimum
    state_in = {
        "messages": case["messages"],
    }
    # Optional extras
    if "conversation_summary" in case:
        state_in["conversation_summary"] = case["conversation_summary"]
    if "user_context_info" in case:
        state_in["user_context_info"] = case["user_context_info"]

    return graph.invoke(state_in)


def _assert_allowed(value: str, allowed: set, field: str) -> None:
    assert value in allowed, f"{field} must be one of {sorted(allowed)}, got: {value!r}"


@pytest.mark.parametrize("case_path", _list_case_files())
def test_schema_contract_hard_fail(case_path: Path, llm):
    """Hard-fail schema contract:
    - required keys exist after graph run
    - routing-critical fields use allowed vocab
    - types match broad expectations
    """
    case = _load_json(case_path)
    case_id = case.get("case_id", case_path.stem)

    run_id = os.environ.get("INTENT_EVAL_RUN_ID", "local")

    out = _run_intake_graph(llm, case)

    write_artifact(ARTIFACTS_DIR, run_id, case_id, "input", case)
    write_artifact(ARTIFACTS_DIR, run_id, case_id, "final_state", out)

    # --- Hard fail on node errors if present ---
    errors = out.get("errors") or []
    assert errors == [], f"Graph returned errors: {errors}"

    # --- Required output keys ---
    # From normalize_gate
    assert "normalized_query" in out and isinstance(out["normalized_query"], str) and out["normalized_query"].strip()
    assert "constraints" in out and isinstance(out["constraints"], dict)
    assert "guardrails" in out and isinstance(out["guardrails"], dict)
    assert "clarification" in out and isinstance(out["clarification"], dict)

    # From extract_signals
    assert "user_intent" in out and isinstance(out["user_intent"], str)
    assert "retrieval_intent" in out and isinstance(out["retrieval_intent"], str)
    assert "answerability" in out and isinstance(out["answerability"], str)
    assert "complexity_flags" in out and isinstance(out["complexity_flags"], list)
    assert "signals" in out and isinstance(out["signals"], dict)

    # --- Allowed vocab checks (routing-critical) ---
    allowed_user_intent = {
        "explain",
        "lookup",
        "compare",
        "decide",
        "troubleshoot",
        "summarize",
        "extract",
        "draft",
        "plan",
        "other",
    }
    allowed_retrieval_intent = {
        "none",
        "definition",
        "procedure",
        "evidence",
        "examples",
        "verification",
        "background",
        "mixed",
    }
    allowed_answerability = {"internal_corpus", "external", "user_context", "reasoning_only", "mixed"}

    _assert_allowed(out["user_intent"], allowed_user_intent, "user_intent")
    _assert_allowed(out["retrieval_intent"], allowed_retrieval_intent, "retrieval_intent")
    _assert_allowed(out["answerability"], allowed_answerability, "answerability")

    # complexity flags - allow empty, but if present must be from known set
    allowed_complexity_flags = {
        "multi_intent",
        "multi_domain",
        "requires_synthesis",
        "requires_strict_precision",
        "long_query",
    }
    for f in out["complexity_flags"]:
        assert f in allowed_complexity_flags, f"Unknown complexity flag: {f!r}"

    # clarification shape (if present)
    clar = out["clarification"]
    if "needed" in clar:
        assert isinstance(clar["needed"], bool)
    if "blocking" in clar:
        assert isinstance(clar["blocking"], bool)
    if "reasons" in clar:
        assert isinstance(clar["reasons"], list)
