# Evaluation Architecture

This project uses a pragmatic evaluation setup based on `pytest` and local artifacts on disk. The goal is to enforce contracts early (schema + key behaviors) while keeping iteration fast and reviewable before investing in heavier observability (e.g., LangSmith, dashboards).

## 🎯 Goals

- **Catch breakages early**: Detect schema shifts and prompt/template regressions before they reach production.
- **Lock down stable behaviors**: Use behavior contracts for high-signal responses.
- **Track drift over time**: Implement soft failures for stability tracking while prompts evolve.
- **Reviewable Artifacts**: Produce on-disk state for manual review and for bootstrapping expected labels.

**Non-goals (for now):**
- Full online evaluation pipelines or hosted experiment tracking.
- Large-scale statistical evaluation (can be added later with Ragas/deepeval).

---

## 🏗️ Test Categories

Tests live under: `tests/intent_eval/`

### 1. `test_schema_contract.py` (**Hard Fail**)
**Purpose:**
- Validate the graph runs end-to-end for each test case.
- Ensure required top-level keys exist in the resulting state.
- Validate that data types match the schema and are non-empty where required.
- Ensure no errors were recorded during execution (`errors == []`).

**Assertions:**
- **Presence of keys**: `normalized_query`, `constraints`, `guardrails`, `clarification`, `user_intent`, `retrieval_intent`, `answerability`, `complexity_flags`, `signals`.
- **Typing**: `normalized_query` is a non-empty string; `constraints`, `guardrails`, `clarification`, and `signals` are dictionary-like; `complexity_flags` is a list.

### 2. `test_behavior_contract.py` (**Hard Fail**)
**Purpose:**
- Assert a small, curated set of stable behaviors on reviewed test cases.
- Act as "guardrails" against prompt drift.
- Only assert fields you intentionally want to remain stable.

**Typical Assertions:**
- `normalized_query_contains`: Substrings that must appear in the normalized output.
- `constraints_format_contains`: Specific subsets like `["no_code"]`.
- `clarification`: Check `needed` boolean and `reasons_contains`.
- **Core Labels**: `user_intent`, `retrieval_intent`, `answerability`.

### 3. `test_stability.py` (**Soft Fail**)
**Purpose:**
- Measure output drift across repeated runs (same model, same prompt, same input).
- Provides visibility into non-deterministic behavior without blocking CI.

**Key Metrics:**
- Intent label agreement rate across $N$ runs (per case).
- Jaccard overlap for `complexity_flags` and `signals`.
- Presence/absence stability for `clarification.needed`.

---

## 📁 Case and Expected File Layout

We use a simple file-based convention for cases and expectations:

- **Input cases**: `tests/intent_eval/cases/intake_v1/*.json`
- **Expected labels**: `tests/intent_eval/expected/intake_v1/*.expected.json`

### Example
- **Input**: `c001_internal_procedure_no_code.json`
- **Expected**: `c001_internal_procedure_no_code.expected.json`

**Expected File Format Example:**
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
```

---

## 📦 Artifact Storage

All tests and helper utilities write artifacts to: `artifacts/intent_eval/<run_id>/`

### Per-case Artifact Set:
- `<case_id>.input.json`: The original input message(s).
- `<case_id>.final_state.json`: The complete state output from the graph.
- `<case_id>.expected.json`: The asserted values (if a behavior contract exists).

**Why artifacts matter:**
- **Review**: Inspect outputs without needing to rebuild or rerun expensive LLM calls.
- **Bootstrap**: Generate expected labels directly from reviewed and approved outputs.
- **Compare**: Diff outputs across different commits or prompt versions to visualize drift.

---

## 🔄 Helper Workflow

Promoting a new test case to a behavior contract:

1. **Add input file** under `tests/intent_eval/cases/intake_v1/`.
2. **Run the helper script** to execute the intake graph and write output to `artifacts/`.
3. **Review the JSON output** to decide which fields are stable enough to assert.
4. **Create an expected file** with only those assertions in `tests/intent_eval/expected/intake_v1/`.
5. **Run the suite**:
   ```bash
   pytest tests/intent_eval/test_behavior_contract.py -q
   ```

---

## 🚀 Running the Suite

Typical commands for different evaluation needs:

- **Smoke tests**: `pytest tests/intent_eval/test_smoke.py -q`
- **Schema tests**: `pytest tests/intent_eval/test_schema_contract.py -q`
- **Behavior tests**: `pytest tests/intent_eval/test_behavior_contract.py -q`
- **Stability tests**: `pytest tests/intent_eval/test_stability.py -q`

**Recommended flags:**
- `-s`: Print artifacts/debug info during execution.
- `-vv`: Detailed diagnostics for timeouts or retries.
- `--timeout=NN`: Prevents suite hangs if an LLM request gets stuck.

---

## ⚙️ Timeouts and Retries

Because tests call live LLMs:
- Set retry policies on LangGraph nodes carefully (e.g., `max_attempts=1` or `2` for CI).
- Favor transport-level timeouts in the LLM client configuration to fail fast.
- Disable retries in stability tests to ensure you are measuring raw model drift.

---

## 📌 Model Configuration

To keep evaluations meaningful:
- **Pin model names** in fixtures (e.g., `gpt-4o`).
- **Pin temperature** to `0.0`.
- **Bump versions**: When changing a prompt or model, increment the `intake_version` and re-review drift.

---

## ✨ Next Extensions

As the project grows, we will mirror this structure for:
- **Planner Subgraph**: Schema contracts for `PlannerState`, behavior contracts for strategy choices.
- **Executor Subgraph**: Stability tests for retrieval round counts and quality.
- **End-to-End**: Ragas/deepeval scores once execution traces are stable.
