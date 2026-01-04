import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


def json_default(o: Any):
    # LangChain messages (HumanMessage, AIMessage, SystemMessage, etc.)
    if isinstance(o, BaseMessage):
        return {
            "type": o.type,
            "content": o.content,
            "additional_kwargs": getattr(o, "additional_kwargs", {}),
            "response_metadata": getattr(o, "response_metadata", {}),
            "id": getattr(o, "id", None),
            "name": getattr(o, "name", None),
        }

    # Pydantic v2 models
    dump = getattr(o, "model_dump", None)
    if callable(dump):
        return dump()

    # Pydantic v1 models
    dump = getattr(o, "dict", None)
    if callable(dump):
        return dump()

    # Last resort: keep artifacts usable
    return str(o)


def write_artifact(base_dir: Path, run_id: str, filename: str, payload: Any) -> Path:
    out_dir = base_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
    return out_path
