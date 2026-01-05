"""Shared utilities for running test cases."""

import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_case(path: Path) -> Dict[str, Any]:
    """Load a single case from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cases(case_pattern: str) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Resolve case pattern to a list of (path, case_data) tuples.

    Supports:
    - Single file: /path/to/case.json
    - Glob pattern: /path/to/case_*.json or tests/*/cases/**/*.json

    Returns list of (case_path, case_data) tuples, sorted by path.
    """
    # Check if it's a direct file (no glob characters)
    if not any(c in case_pattern for c in ['*', '?', '[', ']']):
        path = Path(case_pattern)
        if not path.exists():
            raise FileNotFoundError(f"Case file not found: {case_pattern}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {case_pattern}")
        return [(path, load_case(path))]

    # It's a glob pattern
    matches = sorted(glob(case_pattern, recursive=True))

    if not matches:
        raise FileNotFoundError(f"No cases found matching pattern: {case_pattern}")

    results = []
    for match_str in matches:
        path = Path(match_str)
        if path.is_file() and path.suffix == '.json':
            try:
                case_data = load_case(path)
                results.append((path, case_data))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Skipping {path}: {e}")
                continue

    if not results:
        raise FileNotFoundError(f"No valid JSON case files found matching: {case_pattern}")

    return results


def get_case_id(case_path: Path, case_data: Dict[str, Any]) -> str:
    """Extract case ID from case data or derive from filename."""
    return case_data.get("case_id", case_path.stem)
