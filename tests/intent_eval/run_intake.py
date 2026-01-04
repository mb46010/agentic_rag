# tests/intent_eval/run_intake.py

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_rag.intent.graph import make_intake_graph


def load_case(case_path: Path) -> Dict[str, Any]:
    with case_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_state_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal state builder for the intake graph.
    Expects case["messages"] as a list of chat messages (LangChain message dicts or BaseMessages).
    """
    if "messages" not in case:
        raise ValueError("Case is missing required key: 'messages'")

    state_in: Dict[str, Any] = {"messages": case["messages"]}

    # Optional extras
    if "conversation_summary" in case:
        state_in["conversation_summary"] = case["conversation_summary"]
    if "user_context_info" in case:
        state_in["user_context_info"] = case["user_context_info"]
    if "user_email" in case:
        state_in["user_email"] = case["user_email"]

    return state_in


def run_intake(llm, case: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Run the compiled intake graph against a single case dict.
    Returns the final IntakeState (dict).
    """
    graph = make_intake_graph(llm, max_retries=max_retries)
    state_in = build_state_from_case(case)
    return graph.invoke(state_in)


def run_intake_from_file(llm, case_path: Path, max_retries: int = 3) -> Dict[str, Any]:
    case = load_case(case_path)
    return run_intake(llm, case, max_retries=max_retries)
