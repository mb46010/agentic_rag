# tests/intent_eval/conftest.py

import os
import uuid
from pathlib import Path
from typing import Iterator, Optional
from dotenv import load_dotenv
import pytest

from agentic_rag.model import get_default_model

ARTIFACTS_DIR = Path("artifacts/intent_eval")

# Load .env once for the whole test session
if os.getenv("PYTEST_CURRENT_TEST"):
    load_dotenv()

def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / ".gitkeep").touch(exist_ok=True)


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None

§
@pytest.fixture(scope="session", autouse=True)
def _session_setup() -> None:
    _ensure_artifacts_dir()


@pytest.fixture(scope="session")
def run_id() -> str:
    """Unique id for this pytest run. Used by tests to write artifacts under artifacts/intent_eval/<run_id>/.
    You can override with INTENT_EVAL_RUN_ID for stable paths in CI.
    """
    rid = _env("INTENT_EVAL_RUN_ID")
    if rid:
        return rid
    return f"pytest-{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="session", autouse=True)
def _set_run_id_env(run_id: str) -> None:
    """Tests in this suite look for INTENT_EVAL_RUN_ID to decide artifact directory."""
    os.environ["INTENT_EVAL_RUN_ID"] = run_id


@pytest.fixture(scope="session")
def llm():
    """Default LLM fixture for intent evaluation tests.
    Uses the same model configuration as production.
    """
    return get_default_model()


@pytest.fixture(scope="session")
def llm_alternative():
    """LLM fixture for intent-eval tests.

    Supports:
    - Azure OpenAI via langchain-openai's AzureChatOpenAI
      Requires: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
      Optional: AZURE_OPENAI_API_VERSION

    - OpenAI via langchain-openai's ChatOpenAI
      Requires: OPENAI_API_KEY
      Optional: OPENAI_MODEL (defaults to "gpt-4o-mini")

    If neither is configured, tests are skipped.

    Notes:
    - Temperature is set to 0 for stability.
    - Keep model selection explicit in CI (pin versions).
    """
    try:
        # langchain-openai package
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except Exception as e:
        pytest.skip(f"langchain_openai not available: {e}")

    # Prefer Azure if configured
    azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")
    azure_api_key = _env("AZURE_OPENAI_API_KEY")
    azure_deployment = _env("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = _env("AZURE_OPENAI_API_VERSION") or "2024-06-01"

    if azure_endpoint and azure_api_key and azure_deployment:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=0,
        )

    # Fallback to OpenAI if configured
    openai_api_key = _env("OPENAI_API_KEY")
    if openai_api_key:
        model = _env("DEFAULT_LLM_MODEL") or "gpt-4o-mini"
        return ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0,
        )

    pytest.skip(
        "No LLM credentials found. Set Azure env vars (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_DEPLOYMENT) or OPENAI_API_KEY."
    )
