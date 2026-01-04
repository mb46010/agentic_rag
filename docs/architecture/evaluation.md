# Evaluation Architecture

This project uses a pragmatic evaluation setup based on `pytest` + local artifacts on disk. The goal is to enforce contracts early (schema + key behaviors) while keeping iteration fast and reviewable before investing in heavier observability (Langfuse, dashboards, etc.).

## Goals

- Catch breakages early (schema and prompt/template regressions)
- Lock down a small set of stable, high-signal behaviors
- Track drift over time without blocking progress (soft failures)
- Produce artifacts for review and for creating expected labels

Non-goals (for now)
- Full online eval pipelines or hosted experiment tracking
- Large-scale statistical evaluation (later with Ragas/deepeval if needed)

## Test categories (exact)

Tests live under: `tests/intent_eval/`

### 1) `test_schema_contract.py` (hard fail)

Purpose:
- Validate the graph runs end-to-end for each case
- Validate required top-level keys exist
- Validate types are correct and non-empty where required
- Validate `errors == []`

What it should assert:
- Presence of keys like:
  - `normalized_query`, `constraints`, `guardrails`, `clarification`
  - `user_intent`, `retrieval_intent`, `answerability`, `complexity_flags`, `signals`
- Types:
  - `normalized_query` is non-empty string
  - `constraints`, `guardrails`, `clarification`, `signals` are dict-like
  - `complexity_flags` is list
- Optional but recommended:
  - `intake_version` exists and matches expected prefix

This test should not assert semantic correctness beyond minimal invariants.

### 2) `test_behavior_contract.py` (hard fail, small labeled subset)

Purpose:
- Assert a SMALL, curated set of stable behaviors on reviewed cases
- Only assert fields you intentionally want to keep stable
- Make these tests the "guardrails" against prompt drift

Typical assertions (when present in expected file):
- `normalized_query_contains`: list of substrings that must appear
- `constraints_format_contains`: expected subset like `["no_code"]`
- `clarification.needed` and `clarification.reasons_contains`
- core labels:
  - `user_intent`
  - `retrieval_intent`
  - `answerability`

Important rule:
- Expected files should include only what you want to assert.
- Avoid asserting every extracted entity/acronym early, those drift a lot.

### 3) `test_stability.py` (soft fail initially)

Purpose:
- Measure output drift across repeated runs (same model, same prompt)
- Non-blocking at first while prompts are still evolving
- Later can be turned into hard fail with thresholds

Typical metrics:
- Intent label agreement rate across N runs (per case)
- Jaccard overlap for `complexity_flags`
- Presence/absence stability for `clarification.needed`
- (optional) normalized_query similarity checks (contains-based, not exact)

Recommended approach:
- Run each case `N=3..5` times
- Save all outputs to artifacts
- Compute drift metrics and compare to thresholds
- Initially report-only; later gate merges

## Case and expected file layout

A simple convention:

- Input cases:
  - `tests/intent_eval/cases/intake_v1/*.json`

- Expected labels (for behavior contracts):
  - `tests/intent_eval/expected/intake_v1/*.expected.json`

Suggested naming:
- input: `c001_internal_procedure_no_code.json`
- expected: `c001_internal_procedure_no_code.expected.json`

Each input case should include:
- `case_id`: stable identifier
- `messages`: the minimal messages list

Each expected file should include only asserted keys, for example:

```json
{
  "constraints_format_contains": ["no_code"],
  "clarification": {
    "needed": false
  },
  "user_intent": "plan",
  "retrieval_intent": "procedure",
  "answerability": "internal_corpus"
}

Artifact storage (local, review-first)

All tests and helper utilities write artifacts to:

artifacts/intent_eval/<run_id>/...

Suggested per-case artifact set:

<case_id>.input.json

<case_id>.final_state.json

<case_id>.expected.json (if behavior contract exists)

Artifacts should be JSON-serializable:

Convert LangChain messages to dicts via a JSON default function

Avoid dumping raw objects

Why artifacts matter:

You can review outputs without rerunning

You can generate expected labels from reviewed outputs

You can compare drift across commits

Helper workflow: promote a new case to behavior contract

Recommended loop:

Add input file under:

tests/intent_eval/cases/intake_v1/

Run the helper that:

executes the intake graph

writes output to artifacts

Review output JSON:

decide which fields are stable enough to assert

Create an expected file with only those assertions:

tests/intent_eval/expected/intake_v1/<case>.expected.json

Run:

pytest tests/intent_eval/test_behavior_contract.py -q

If the model sometimes flips labels, reduce assertions to more robust ones:

use *_contains lists instead of exact strings

assert clarification.needed but not the full reasons list

assert subsets not full lists

Running the suite

Typical commands:

smoke only:

pytest tests/intent_eval/test_smoke.py -q

schema contracts only:

pytest tests/intent_eval/test_schema_contract.py -q

behavior contracts:

pytest tests/intent_eval/test_behavior_contract.py -q

stability:

pytest tests/intent_eval/test_stability.py -q

Recommended flags:

-s when debugging prompts / printing artifacts

-vv when diagnosing timeouts and retries

--timeout=NN if you use pytest-timeout for stuck requests

Timeouts and retries

Because tests call an LLM:

set retry policy on LangGraph nodes carefully

consider disabling retries in tests to avoid long hangs

Practical guidance:

for schema and behavior tests:

max_attempts=1 or 2 (keep fast and deterministic)

for stability tests:

max_attempts=1 (you want to measure drift, not recovery)

Also prefer setting per-request timeouts in the LLM client configuration (transport-level), so failures are fast and explicit.

Model/config pinning

To keep evals meaningful:

pin model name in fixture (example: gpt-4.1)

pin temperature to 0.0

keep max_tokens high enough to avoid truncation causing schema failures

When changing model or prompt versions:

bump intake_version

run schema + behavior tests and re-review drift

Optional: disabling unwanted pytest plugins

Some dependencies may pull in plugins (example: langsmith).
To avoid side effects:

disable via pytest.ini:



[pytest]
addopts = -v --strict-markers -p no:langsmith

This prevents plugin hooks from affecting runtime, logging, or network usage.

Next extensions

When you add the planner and executor:

mirror the same test structure per component:

schema contract for PlannerState

behavior contract for plan choices and stop conditions

stability tests for plan drift and round counts

Later, when retrieval is real:

add retrieval correctness tests with mocked adapters

add end-to-end evals (Ragas/deepeval) only after you have stable execution traces

::contentReference[oaicite:0]{index=0}
