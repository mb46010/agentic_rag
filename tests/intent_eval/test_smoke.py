# tests/intent_eval/test_smoke.py
"""Smoke test for the intake graph.

Runs the intake graph end-to-end on a minimal input and checks that it returns the expected top-level keys.

Usage:
    pytest tests/intent_eval/test_smoke.py
"""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from dotenv import load_dotenv

from agentic_rag.intent.graph import make_intake_graph

load_dotenv(".env", override=False)

ARTIFACTS_DIR = Path("artifacts/intent_eval")


import json
from pathlib import Path

from json_utils import write_artifact
from langchain_core.messages import BaseMessage


def test_intake_graph_smoke(llm):
    """Dry-run smoke test:
    - Graph compiles
    - Graph runs end-to-end on a minimal input
    - Returns required top-level keys
    - No errors in output state
    """
    graph = make_intake_graph(llm, max_retries=2)

    state_in: Dict[str, Any] = {
        "messages": [{"role": "user", "content": "Summarize the main point of our HR policy on vacations. No code."}]
    }

    out = graph.invoke(state_in)

    run_id = os.environ.get("INTENT_EVAL_RUN_ID", "local")
    write_artifact(ARTIFACTS_DIR, run_id, "smoke", "input", state_in)
    write_artifact(ARTIFACTS_DIR, run_id, "smoke", "output", out)

    errors = out.get("errors") or []
    assert errors == [], f"Smoke test failed with errors: {errors}"

    # Minimal required outputs after full intake graph
    assert isinstance(out.get("normalized_query"), str) and out["normalized_query"].strip()
    assert isinstance(out.get("constraints"), dict)
    assert isinstance(out.get("guardrails"), dict)
    assert isinstance(out.get("clarification"), dict)

    assert isinstance(out.get("user_intent"), str) and out["user_intent"].strip()
    assert isinstance(out.get("retrieval_intent"), str) and out["retrieval_intent"].strip()
    assert isinstance(out.get("answerability"), str) and out["answerability"].strip()
    assert isinstance(out.get("complexity_flags"), list)
    assert isinstance(out.get("signals"), dict)
