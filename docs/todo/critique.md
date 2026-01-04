# Agentic RAG - Code Critique

**Date:** 2026-01-04
**Scope:** src/agentic_rag codebase review

---

## Executive Summary

The codebase demonstrates strong architectural principles with clean separation of concerns, well-defined contracts, and good use of modern Python patterns (Pydantic, LangGraph, Protocol-based adapters). However, there are implementation gaps between the documented architecture and the actual code, plus several low-effort improvements that would significantly boost robustness and maintainability.

---

## Critical Issues

### 1. Missing Core Components

**Issue:** Architecture documents describe 4 subgraphs/nodes (Intake, Planner, Executor, Answer), but only Intake and Executor are implemented.

**Impact:** High - core functionality incomplete

**Files affected:**
- Missing: `src/agentic_rag/planner/`
- Missing: `src/agentic_rag/answer/`

**Recommendation:**
- Stub out planner and answer nodes with basic implementations
- OR update architecture docs to mark these as "planned" rather than implemented


### 2. Missing Package Initialization Files

**Issue:** No `__init__.py` files in package directories

**Impact:** Medium - makes imports fragile and IDE support poor

**Files needed:**
- `src/agentic_rag/__init__.py`
- `src/agentic_rag/intent/__init__.py`
- `src/agentic_rag/intent/nodes/__init__.py`
- `src/agentic_rag/intent/prompts/__init__.py`
- `src/agentic_rag/executor/__init__.py`
- `src/agentic_rag/executor/nodes/__init__.py`

**Recommendation:** Add minimal `__init__.py` files to all package directories

**Effort:** Low | **Impact:** Medium


### 3. Test Coverage Gap

**Issue:** Only intent evaluation tests exist. No tests for executor graph or individual executor nodes.

**Impact:** High - executor is untested and likely has bugs

**Files affected:**
- No tests for: `executor/graph.py`, `executor/nodes/*.py`

**Recommendation:**
- Add unit tests for executor nodes with mocked adapters
- Add integration test for executor graph with stub adapters

**Effort:** Medium | **Impact:** High

---

## Low-Effort, High-Impact Improvements

### 1. Add Missing intake_version Field ⭐

**Issue:** Documentation specifies `intake_version` output field (architecture_intent.md:219, evaluation.md:37), but normalize_gate doesn't set it.

**Location:** `src/agentic_rag/intent/nodes/normalize_gate.py:100-113`

**Fix:**
```python
# In normalize_gate, add to output dict:
out["intake_version"] = "intake_v1"  # or read from NORMALIZE_PROMPT_VERSION
```

**Effort:** 1 line | **Impact:** High (enables version tracking)


### 2. Add debug_notes Field ⭐

**Issue:** Documentation specifies `debug_notes` output (architecture_intent.md:220), but it's never populated.

**Location:** `src/agentic_rag/intent/nodes/extract_signals.py:145-151`

**Fix:**
```python
# Add debug_notes to ExtractSignalsModel if useful, or set to empty string in output
return {
    # ... existing fields ...
    "debug_notes": "",  # or model.debug_notes if added to schema
}
```

**Effort:** Low | **Impact:** Low (but improves contract compliance)


### 3. Make Candidate Immutable ⭐⭐

**Issue:** Candidate dataclass is mutable, but nodes mutate it in-place (run_retrieval.py:46-48), which can cause subtle bugs.

**Location:** `src/agentic_rag/executor/state.py:119-139`

**Fix:**
```python
@dataclass(frozen=True)  # Add frozen=True
class Candidate:
    # ... existing fields ...
```

Then update run_retrieval to create new instances instead of mutating:
```python
# Instead of: h.round_id = ...; h.query = ...; h.mode = ...
# Use dataclasses.replace:
from dataclasses import replace
for h in hits:
    raw.append(replace(h, round_id=..., query=q, mode=mode))
```

**Effort:** Low | **Impact:** High (prevents mutation bugs)


### 4. Extract Magic Numbers to Constants ⭐

**Issue:** Hardcoded values scattered throughout (60, 200, 12, etc.)

**Locations:**
- `executor/nodes/merge_candidates.py:62` - rrf k=200, rrf_k=60
- `executor/nodes/finalize_evidence_pack.py:18` - max_total_docs=12
- `executor/nodes/select_evidence.py:48` - max_docs=8
- `executor/nodes/rerank_candidates.py:24` - rerank_top_k=60

**Fix:** Create `executor/constants.py`:
```python
# executor/constants.py
DEFAULT_RRF_K = 60
DEFAULT_RRF_POOL_SIZE = 200
DEFAULT_MAX_TOTAL_DOCS = 12
DEFAULT_MAX_DOCS_PER_ROUND = 8
DEFAULT_RERANK_TOP_K = 60
```

**Effort:** Low | **Impact:** Medium (improves maintainability)


### 5. Add Plan Validation in Executor Gate ⭐⭐

**Issue:** executor_gate assumes plan structure but doesn't validate it thoroughly. Missing fields cause silent failures.

**Location:** `src/agentic_rag/executor/nodes/executor_gate.py:10-68`

**Fix:**
```python
def executor_gate(state: ExecutorState) -> Dict[str, Any]:
    plan = state.get("plan")
    if not plan:
        return {
            "errors": [{
                "node": "executor_gate",
                "type": "schema_validation",
                "message": "Missing plan in ExecutorState",
                "retryable": False,
                "details": None,
            }],
            "continue_search": False,
        }

    strategy = plan.get("strategy")
    if not strategy:
        return {"errors": [...], "continue_search": False}

    # ... rest of validation ...
```

**Effort:** Low | **Impact:** High (prevents cryptic errors downstream)


### 6. Consistent Error Handling Pattern ⭐

**Issue:** Some nodes return errors dict, some might throw exceptions. Inconsistent pattern.

**Locations:** All executor nodes

**Fix:** Wrap all node bodies in try/except and always return error dicts:
```python
def node_function(state: ExecutorState) -> Dict[str, Any]:
    try:
        # ... node logic ...
        return {"field": value}
    except Exception as e:
        return {
            "errors": [{
                "node": "node_name",
                "type": "runtime_error",
                "message": str(e),
                "retryable": True,
                "details": {"exception_type": type(e).__name__},
            }]
        }
```

**Effort:** Low | **Impact:** High (improves reliability)


### 7. Add @observe Decorator Consistently ⭐

**Issue:** Only intent nodes use @observe, executor nodes don't. Inconsistent observability.

**Locations:** All executor nodes lack @observe

**Fix:** Add the same observe pattern from intent nodes to all executor nodes:
```python
from agentic_rag.intent.nodes.normalize_gate import OBSERVE_ENABLED, observe

@observe
def node_function(state: ExecutorState) -> Dict[str, Any]:
    # ... existing logic ...
```

**Effort:** Low | **Impact:** Medium (improves debugging)


### 8. Add Logging Statements ⭐

**Issue:** No logging in executor nodes. Makes debugging difficult.

**Locations:** All executor nodes

**Fix:**
```python
import logging
logger = logging.getLogger(__name__)

def node_function(state: ExecutorState) -> Dict[str, Any]:
    logger.info("Starting node_name with round_index=%s", state.get("current_round_index", 0))
    # ... logic ...
    logger.debug("Completed node_name: selected=%d items", len(selected))
    return result
```

**Effort:** Low | **Impact:** Medium (improves debugging)


### 9. Type Safety: Use Actual Types Instead of Dict[str, Any] ⭐⭐

**Issue:** IntakeState and ExecutorState use `Dict[str, Any]` for constraints, guardrails, signals. Loses type safety.

**Locations:**
- `intent/state.py:196-201`
- `executor/state.py:176-178`

**Fix:**
```python
# In IntakeState, change from:
constraints: Dict[str, Any]
# To:
constraints: Constraints  # Use the actual TypedDict

# Same for guardrails, signals, etc.
```

**Effort:** Low | **Impact:** High (catches bugs at dev time)


### 10. Document Adapter Implementations ⭐

**Issue:** Adapter protocols are well-defined, but no example implementations or documentation on how to implement them.

**Location:** `executor/adapters.py`

**Fix:** Add docstring examples to each Protocol:
```python
class RetrieverAdapter(Protocol):
    """Adapter for retrieval backends.

    Example implementation:
        class AzureSearchRetriever:
            def search(self, *, query, mode, k, alpha, filters):
                # Call Azure AI Search API
                # Map results to Candidate objects
                return [Candidate(...), ...]
    """
```

**Effort:** Low | **Impact:** Medium (improves developer experience)

---

## Medium-Effort Improvements

### 11. Add Prompt Version Tracking

**Issue:** Prompts have version constants but they're not used or logged.

**Locations:**
- `intent/prompts/normalize.py:101`
- `intent/prompts/extract_signals.py:141`

**Fix:** Log prompt versions in nodes, include in intake_version field.

**Effort:** Medium | **Impact:** Medium


### 12. Improve Diversity Selection Algorithm

**Issue:** `_diverse_top_k` in select_evidence.py uses simple round-robin. Could be smarter about entity coverage.

**Location:** `executor/nodes/select_evidence.py:11-34`

**Fix:** Use signals.entities and acceptance_criteria.must_cover_entities to prioritize chunks that cover missing entities.

**Effort:** Medium | **Impact:** Medium


### 13. Add Telemetry/Metrics

**Issue:** No metrics collection (latency, token usage, retrieval counts, etc.)

**Fix:** Add structured telemetry throughout the graph:
- Intake latency
- Retrieval call count and latency per round
- Token usage per LLM call
- Final evidence pool size

**Effort:** Medium | **Impact:** Medium (enables monitoring)


### 14. Validate Literal Type Constraints at Runtime

**Issue:** Literals defined in state.py (UserIntent, RetrievalIntent, etc.) but LLM can return invalid values. Only test_schema_contract catches this.

**Fix:** Add Pydantic validators or Literal checks in node outputs.

**Effort:** Medium | **Impact:** Medium


### 15. Add Retry Logic for Transient Failures

**Issue:** LangGraph retry_policy is set, but individual adapter calls might need circuit breakers for external services.

**Fix:** Add tenacity or similar retry logic around retriever.search, reranker.rerank calls.

**Effort:** Medium | **Impact:** Medium

---

## Code Quality Issues

### Inconsistent Naming

- Some nodes use `state.get("field")`, others assume fields exist
- Some use `List[Candidate]`, others use `list` - be consistent
- Error dict keys: sometimes `"details": None`, sometimes omitted

### Missing Docstrings

Most node functions lack docstrings. Add at minimum:
- Purpose
- Inputs from state
- Outputs to state
- Side effects (if any)

### Type Annotations

Some functions missing return type hints:
- `_preserve_literal_terms` in prepare_round_queries.py
- `_dedupe` in merge_candidates.py
- `_diverse_top_k` in select_evidence.py

---

## Security & Safety

### Potential Issues

1. **No input sanitization** - user messages passed directly to LLM prompts. Consider prompt injection risks.

2. **No rate limiting** - retriever/reranker calls in loops could DoS external services.

3. **No PII filtering** - `pii_present` flag is set in guardrails but never acted upon. Should redact or refuse if PII detected.

4. **Unbounded loops** - executor loop has max_rounds check but no global timeout. Could run indefinitely if logic bug.

### Recommendations

- Add input validation and sanitization before LLM calls
- Add circuit breakers/rate limiters to adapter calls
- Implement PII filtering if `guardrails.pii_present == true`
- Add global timeout to executor graph invocation

---

## Testing Recommendations

### Missing Test Scenarios

1. **Executor unit tests** - test each node in isolation with mocked adapters
2. **Error propagation tests** - verify errors flow correctly through graph
3. **State contract tests** - verify each node only writes fields it owns
4. **Boundary condition tests** - empty results, max limits, null fields
5. **Integration tests** - full graph with stub implementations

### Test Infrastructure Improvements

- Add factory functions for creating test states
- Add fixtures for common adapter mocks
- Add helper to assert error structure
- Add snapshot testing for prompt outputs

---

## Documentation-Code Mismatches

1. **intake_version** - documented but not set (see improvement #1)
2. **debug_notes** - documented but not set (see improvement #2)
3. **Planner node** - documented but not implemented
4. **Answer node** - documented but not implemented
5. **language/locale** - returned by normalize_gate but never used downstream

---

## Priority Recommendations

**Do First (Low Effort, High Impact):**
1. ⭐⭐⭐ Add `__init__.py` files to all packages
2. ⭐⭐⭐ Add `intake_version` field to normalize_gate output
3. ⭐⭐⭐ Make Candidate frozen/immutable
4. ⭐⭐ Add plan validation to executor_gate
5. ⭐⭐ Add error handling wrappers to all executor nodes

**Do Next (Medium Effort, High Impact):**
6. Add executor node tests with mocked adapters
7. Extract magic numbers to constants
8. Add type safety (use Constraints/Guardrails types instead of Dict)
9. Add consistent @observe and logging
10. Add adapter implementation examples/documentation

**Consider Later:**
- Implement planner node (or remove from docs)
- Implement answer node (or remove from docs)
- Add telemetry/metrics collection
- Improve diversity selection algorithm
- Add PII filtering enforcement

---

## Conclusion

The codebase has a solid foundation with excellent architectural principles. The main issues are:
1. **Incomplete implementation** (missing planner/answer nodes)
2. **Missing tests** for executor
3. **Type safety gaps** (Dict[str, Any] instead of typed structures)
4. **Inconsistent patterns** (error handling, observability)

Addressing the 10 low-effort improvements would significantly boost code quality and robustness with minimal time investment. Most are 1-5 line changes.

The architecture is sound - focus on implementation consistency and test coverage.
