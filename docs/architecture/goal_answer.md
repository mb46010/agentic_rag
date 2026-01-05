Here’s a clean requirements sketch for the **Answer graph** (post-executor). It should be boring, strict, and easy to test.

---

## 1) Answer graph role

The answer graph answers only:

> “Given selected evidence + the plan’s answer requirements, produce the final user response (or a clarification/refusal), without inventing anything.”

It does **not**:

* do retrieval
* rerank
* expand the evidence set
* decide search strategy (planner already did)

---

## 2) Inputs (contract)

Minimum required inputs from upstream state:

From **IntakeState**:

* `messages`
* `constraints` (format, prohibitions like no_code)
* `guardrails` (sensitivity, pii)
* `language`, `locale` (optional)

From **PlannerState** (planner output):

* `plan.goal`
* `plan.strategy`
* `plan.answer_requirements` (format, tone, length, citation_style)
* `plan.safety` (sensitivity, pii_allowed)
* `plan.acceptance_criteria` (helps decide if missing coverage is acceptable)

From **Executor outputs**:

* `final_evidence` (list of chunks with provenance + scores)
* `coverage` (covered/missing + confidence + contradictions)
* `retrieval_report` (optional, for internal debug only)

---

## 3) Outputs (contract)

The answer stage should write:

* `final_answer: str` (the only user-facing content)
* `citations: List[Citation]` (structured, if you support citations)
* `answer_meta`:

  * `answer_version`
  * `used_evidence_ids`
  * `coverage_confidence`
  * `refusal: bool`
  * `asked_clarification: bool`

Optional but useful:

* `followups: List[str]` (non-blocking suggested next questions)
* `errors: List[...]` (same pattern as other graphs)

---

## 4) Evidence contract (what the answerer is allowed to use)

Define a strict evidence schema (even if executor already has one). At minimum each item:

* `evidence_id` (stable)
* `content` (text snippet)
* `source` (doc title or URI)
* `doc_id`, `chunk_id`
* `metadata` (section, timestamp, etc.)
* `scores` (optional)
* `provenance` (round_id, query, mode)

Hard rule:

* The answer model may only assert facts that are supported by at least one evidence snippet.
* If evidence is missing: it must say so or ask a clarification.

---

## 5) Key behaviors and gates

### A) Safety + policy gate (pre-answer)

Inputs:

* intake `guardrails`, planner `safety`, user constraints

Decisions:

* If `sensitivity == restricted` and request is not allowed -> refuse or ask user to re-scope.
* If `pii_present == true` and `pii_allowed == false` -> redact or refuse depending on your org policy.
* If user prohibits certain actions (no_external, no_citations) comply.

Output:

* either proceed
* or `final_answer` is a refusal/redirect with safe alternative.

### B) Coverage gate (answer vs clarify)

Use executor `coverage` + plan acceptance criteria:

If any of these:

* missing required entities/subquestions
* confidence below threshold (if planner set one)
* contradictions unresolved
  then:
* either ask clarification (if the missing piece is user-provided)
* or state limitations (if missing requires more corpus)

This is where “deep search agent” feels reliable: it does not bluff.

### C) Citation policy

If answer_requirements asks for citations (or default policy requires it):

* include citations aligned to evidence IDs
* do not cite things not in evidence
* avoid dumping raw URLs if you have a preferred style

If citations are not requested:

* still keep internal mapping `used_evidence_ids` for debugging
* optionally include “According to policy X” style without formal citations

### D) Formatting and UX requirements

Enforce:

* `no_code` (or any format constraints from intake)
* language/locale if needed
* answer length constraints: short/medium/long
* output structure: bullets, steps, table, etc.

---

## 6) Suggested nodes in the Answer graph

Keep it small (2-4 nodes). Example:

1. `answer_gate`

* safety/policy checks
* detect if clarification/refusal is required
* decide answer mode: `answer`, `clarify`, `refuse`

2. `compose_answer`

* LLM call with strict prompt:

  * goal
  * selected evidence snippets
  * formatting requirements
  * “only use evidence” rules
* output structured: answer text + citations + used evidence IDs

3. `postprocess_answer`

* enforce formatting rules:

  * remove code blocks if forbidden
  * ensure bullet formatting if required
  * citation normalization
* optional light deterministic rewriting, not content changes

4. `finalize`

* write final fields to state

---

## 7) Prompt requirements for `compose_answer`

The prompt must include:

* the plan goal
* explicit constraints: “do not add facts not supported”
* evidence list with IDs
* what to do when evidence is insufficient:

  * ask 1-3 clarifying questions OR state limitation
* required format and tone
* citation requirements: style and mapping

Make it non-creative:

* temperature 0
* structured output (function_calling) returning:

  * `final_answer`
  * `citations`
  * `used_evidence_ids`
  * `answered_subquestions` (optional)
  * `missing_info` (optional)

---

## 8) Evaluation tests you can write immediately

Without real retrieval quality:

### Schema contract (hard)

* `final_answer` non-empty string
* if citations requested, citations list exists and references valid evidence_ids

### Behavior contract (hard, curated)

Cases:

1. insufficient coverage -> asks clarification, does not bluff
2. literal term present (error id) -> repeats exact string, no mutation
3. no_code constraint -> no fenced code blocks

### Stability (soft)

* run the same evidence pack 3 times, ensure:

  * answer mode stable (answer vs clarify)
  * citations remain valid
  * does not introduce new entities not in evidence

---

## 9) Practical design constraints (so this stays sane)

* The answer stage should accept evidence packs up to a fixed max tokens; if too large, truncate deterministically:

  * prefer higher rerank score
  * ensure diversity across documents
* Do not pass the entire retrieval report to the answer model (keep internal).
* Keep refusal/clarification templates deterministic and policy-driven.

---

If you want, next step is: define the AnswerState schema (Pydantic) and the `compose_answer` structured output schema so your tests can assert citations and “no bluffing” properties.
