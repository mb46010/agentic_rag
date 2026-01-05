
## 1) Planner’s role (re-stated precisely)

The planner answers **only**:

> “Given the intake signals, how should we search, when should we stop, and what counts as success?”

It does **not**:

* answer the user
* summarize documents
* interpret results

---

## 2) PlannerState (schema / contract)

Think of this as the *output* of the planner node and the *input* of your retrieval executor.

### Core strategy

```text
PlannerState:
  goal: str
  strategy: Literal[
    "direct_answer",
    "retrieve_then_answer",
    "clarify_then_retrieve",
    "defer_or_refuse"
  ]
```

* `goal`: one-sentence restatement, normalized and scoped
* `strategy`: single controlling decision (no mixtures)

---

### Clarification (only populated if needed)

```text
clarifying_questions:
  - question: str
    reason: ClarificationReason
    blocking: bool
```

Rules:

* max 1–3 questions
* each question must unblock a concrete planning uncertainty
* no open-ended “tell me more”

---

### Retrieval plan (the heart of agentic RAG)

```text
retrieval_rounds: List[RetrievalRound]
```

Each round is explicit and small.

```text
RetrievalRound:
  round_id: int
  purpose: Literal[
    "recall",
    "precision",
    "verification",
    "gap_filling"
  ]

  query_variants: List[str]

  retrieval_modes:
    - type: Literal["bm25", "vector"]
      k: int
      alpha: Optional[float]   # only for hybrid setups

  filters:
    doc_types: Optional[List[str]]
    domains: Optional[List[str]]
    entities: Optional[List[str]]
    time_range: Optional[str]

  use_hyde: bool
  rrf: bool

  rerank:
    enabled: bool
    model: Literal["cross_encoder"]
    rerank_top_k: int

  output:
    max_docs: int
```

Key design rules:

* 1–2 rounds by default
* 3 rounds only if `complexity_flags` include `requires_synthesis`
* every round has a **purpose**, not just “search again”

---

### Literal handling (critical for correctness)

```text
literal_constraints:
  must_preserve_terms: List[str]
  must_match_exactly: bool
```

Used for:

* error messages
* IDs
* quoted policy names
* stack traces

This is how you prevent HyDE or semantic search from corrupting exact strings.

---

### Acceptance and stopping rules

```text
acceptance_criteria:
  min_independent_sources: int
  require_authoritative_source: bool
  must_cover_entities: List[str]
  must_answer_subquestions: Optional[List[str]]

stop_conditions:
  max_rounds: int
  max_total_docs: int
  confidence_threshold: Optional[float]
  no_new_information_rounds: int
```

This is what makes it “deep search” instead of “search until bored”.

---

### Output requirements (hand-off to answer node)

```text
answer_requirements:
  format: List[str]        # e.g. ["bullet_points", "citations"]
  tone: Optional[str]
  length: Optional[str]
  citation_style: Optional[str]
```

Planner decides *what kind* of answer, not the content.

---

### Budget and safety

```text
budget:
  max_tokens: int
  max_latency_ms: Optional[int]

safety:
  sensitivity: Sensitivity
  pii_allowed: bool
```

Important for enterprise constraints later.

---

### Traceability (non-reasoning)

```text
planner_meta:
  planner_version: str
  rationale_tags: List[str]   # e.g. ["ambiguous_acronym", "literal_term_present"]
```

These tags are gold for evals and debugging without leaking chain-of-thought.

---

## 3) How intake maps cleanly into planner decisions

Very explicitly:

* `clarification.needed && blocking`
  → `strategy = clarify_then_retrieve`

* `retrieval_intent == none` or `answerability == reasoning_only`
  → `strategy = direct_answer`

* `artifact_flags` or `literal_terms present`
  → no HyDE, keyword-heavy, exact matching

* `complexity_flags.requires_synthesis`
  → 2–3 rounds, explicit acceptance criteria

This makes planner behavior **predictable and testable**.

---

## 4) Planner behavior tests you can write immediately (no infra)

Once planner exists, you can test:

* ambiguous acronym → `clarify_then_retrieve`
* troubleshooting error → `retrieve_then_answer`, `bm25-heavy`, `literal_constraints.must_match_exactly = true`
* conceptual architecture question → `use_hyde = true`, `vector-heavy`
* strict policy lookup → `verification` round, `authoritative_source = true`

These tests do **not** depend on retrieval results. That’s intentional.

---

## 5) One critical warning (learned the hard way)

Do **not** let the planner:

* output prose
* justify decisions in paragraphs
* generate queries implicitly inside text

Everything must be structured. If you feel tempted to add text, add a tag instead.



