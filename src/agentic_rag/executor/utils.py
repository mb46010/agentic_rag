# src/agentic_rag/executor/utils.py
"""Utilities for executor nodes: error handling, observability, logging."""

import functools
import logging
import os
from typing import Any, Callable, Dict

# Observability setup (same pattern as intent nodes)
OBSERVE_ENABLED = os.getenv("LANGFUSE_ENABLED", "1") == "1"

if OBSERVE_ENABLED:
    try:
        from langfuse.decorators import observe
    except ImportError:

        def observe(fn=None, **kwargs):
            def _wrap(f):
                return f

            return _wrap(fn) if fn else _wrap
else:

    def observe(fn=None, **kwargs):
        def _wrap(f):
            return f

        return _wrap(fn) if fn else _wrap


def with_error_handling(node_name: str) -> Callable:
    """Decorator to add consistent error handling to executor nodes.

    Wraps node functions in try/except and returns structured error dicts.
    Also adds logging for debugging.

    Args:
        node_name: Name of the node for error reporting

    Example:
        @with_error_handling("run_retrieval")
        def run_retrieval(state: ExecutorState) -> Dict[str, Any]:
            # ... node logic ...
            return {"round_candidates_raw": raw}
    """

    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                logger.debug(f"Starting {node_name}")
                result = func(state)
                logger.debug(f"Completed {node_name}: {len(result)} fields returned")
                return result
            except Exception as e:
                logger.exception(f"Error in {node_name}: {e}")
                return {
                    "errors": [
                        {
                            "node": node_name,
                            "type": "runtime_error",
                            "message": str(e),
                            "retryable": True,
                            "details": {"exception_type": type(e).__name__},
                        }
                    ]
                }

        return wrapper

    return decorator
