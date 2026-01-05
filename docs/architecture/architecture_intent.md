# Intent Intake Subgraph Architecture

The intent (intake) subgraph provides a stable "first look" at the user request. It extracts normalized inputs and planning signals while keeping this step separate from planning and retrieval.

## Purpose

- **Standardize the request** for logging and downstream planning.
- **Identify risk and sensitivity signals** early (e.g., PII, sensitive topics).
- **Detect ambiguity** and decide whether clarification is blocking.
- **Extract structured signals** that improve later planning and retrieval.

**Intake must not:**
- Retrieve documents.
- Answer the user.
- Generate long reasoning text.

---

## Inputs

### Primary Inputs
- `messages`: Chat history (LangChain message objects or role/content dictionaries).

### Optional Inputs
- `user_context_info`: App-provided context (ACL, organization metadata, user role).
- `conversation_summary`: Optional summary from an upstream system.

---

## Outputs

The intake subgraph writes to `IntakeState`, structured into three main categories:

### 1. Normalize + Gate
- `normalized_query`: A cleaned, minimal query for downstream use.
- `constraints`: Explicit format requirements, prohibitions, domain hints, and non-functional constraints.
- `guardrails`: Time sensitivity, context dependency, sensitivity flags, and PII detection.
- `clarification`: Boolean indicating if more info is needed, including the reasons.

### 2. Signals for Planning
- `user_intent`: Classification (e.g., `explain`, `lookup`, `compare`, `decide`, `troubleshoot`, `summarize`, `extract`, `draft`, `plan`).
- `retrieval_intent`: Type of information sought (e.g., `definition`, `procedure`, `evidence`, `examples`, `verification`, `background`).
- `answerability`: Expected source of Truth (e.g., `internal_corpus`, `external`, `user_context`, `reasoning_only`).
- `complexity_flags`: Indicators like `multi_intent`, `requires_strict_precision`, or `long_query`.
- `signals`: Entities, acronyms, artifact flags, and literal terms.

### 3. Metadata
- `intake_version`: Current version string.
- `debug_notes`: Short, non-CoT notes (avoids chain-of-thought for speed/cost).

---

## Subgraph Overview

The intake subgraph is intentionally small and decomposed:

```mermaid
flowchart LR
  S[START] --> N1[normalize_gate]
  N1 --> N2[extract_signals]
  N2 --> E[END]
```

---

## Node Breakdown

### 1. `normalize_gate`
**Responsibilities:**
- Validate presence and shape of input `messages`.
- Produce `NormalizeModel` output containing:
  - `normalized_query`
  - `constraints`
  - `guardrails`
  - `clarification`
  - Optional: `language`, `locale`.

**Best Practices:**
- Use **structured output** via LLM function calling/tool calling.
- Keep the prompt focused to ensure it remains **cheap and stable**.
- **Do not overfit**: Only normalize and gate; do not attempt to plan or reason.

**Failure Behavior:**
- Write structured errors containing `node`, `type`, `retryable` status, and `details`.
- Avoid throwing exceptions; prefer updating the state with error flags.

### 2. `extract_signals`
**Responsibilities:**
- Classify **intent fields** and **complexity**.
- Extract **signals**: entities, acronyms, artifact flags (stacktraces, paths), and literal terms.
- Use the `normalized_query` from the previous step as context.

**Best Practices:**
- Enforce a **strict schema** (e.g., Pydantic with `extra="forbid"`).
- Use `Literal` types for labels to avoid free-text drift.
- Prefer **conservative defaults** (e.g., empty lists) when signals are not detected.

---

## Design Guidelines

### Keep Intake Stable
Intake outputs are used throughout the system. Any drift here propagates downstream.
- Prefer short prompts and well-defined structured schemas.
- Add fields gradually and only after evaluation.

### Separate "Signal Extraction" from Planning
Intake extracts **what** is in the query, while the Planner decides **how** to solve it. 
- This separation allows you to validate signal quality independently of retrieval results.

### Literal Terms are Critical
If a user message includes error messages, stacktraces, IDs, quoted phrases, file paths, or URLs:
1. **Preserve them** in `signals.literal_terms`.
2. **Set** `requires_strict_precision` in `complexity_flags`.
3. This informs the Execution phase to avoid HyDE and reduce semantic drift for high-precision terms.

---

## Evaluation Targets

### Minimum Quality Checks
- `normalized_query` is never empty for valid requests.
- `constraints.format` correctly includes "no_code" if specifically requested.
- Ambiguous acronyms trigger the `clarification` reason.
- Artifact flags (e.g., `has_stacktrace`) are set correctly for paths/URLs.
- `user_intent` classification matches the project rubric for a reviewed test set.

---

## Known Risks

### 1. Over-Classification
- **Risk:** Everything being classified as `explain`.
- **Mitigation:** Use clear rubric examples in the prompt and enforce `Literal` labels. Add small "contract tests" for specific behavior.

### 2. Prompt Variable Leakage
- **Risk:** Template missing variables or receiving null inputs.
- **Mitigation:** Keep prompt inputs explicit and minimal. Implement smoke tests in CI that run a dummy intent pass.



