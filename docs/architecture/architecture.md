# Agentic RAG Architecture

This project implements an agentic RAG system using LangGraph. The system is decomposed into subgraphs with strict contracts between them to keep behavior testable, auditable, and adaptable to different retrieval backends.

## Goals

- Deterministic orchestration, minimal "magic"
- Clear separation of concerns:
  - Intake: understand the request and extract planning signals
  - Planner: produce an executable search plan
  - Executor: run retrieval, merge/rerank, select evidence, grade coverage, stop
  - Answer: synthesize response using selected evidence and constraints
- Pluggable adapters for RAG infrastructure:
  - Hybrid retrieval (optional alpha)
  - RRF fusion
  - Cross-encoder reranking (CER)
  - HyDE query expansion
- Evaluation-first design: schema contracts, behavior contracts, stability tests

## High-level flow

```mermaid
flowchart LR
  U[User messages] --> I[Intake subgraph]
  I --> P[Planner node]
  P --> E[Executor subgraph]
  E --> A[Answer node]
  A --> U2[Final response]

  I -->|clarification needed| Q[Clarification question]
  Q --> U


# Subgraph responsibilities

## 1) Intake subgraph (intent graph)

Inputs:

- messages
- optional user context info

Outputs:

- normalized query

- constraints + guardrails

- clarification need (including blocking vs non-blocking)

- planning signals (intent labels, entities, acronyms, artifact flags)

Design principle:

- Intake must be stable and cheap
- Intake does not retrieve or answer

See: architecture_intent.md

## 2) Planner node

Inputs:

- intake outputs

Outputs:

PlannerState (an executable plan)

- strategy: direct answer vs retrieve vs clarify

- retrieval rounds: query variants, modes, filters, use_hyde, rrf, rerank

- acceptance criteria and stop conditions

- budget knobs

Design principle:

- Planner output is structured and auditable

- Planner does not write the final answer

## 3) Executor subgraph

Inputs:

- PlannerState

- normalized_query + constraints/guardrails/signals

Outputs:

- final_evidence: selected chunks with provenance and scores

- coverage: what is answered and what is missing

- retrieval_report: round-by-round telemetry for debugging

errors if any

Design principle:

- Executor is mostly deterministic

- LLM is used only for bounded tasks (HyDE, coverage grading)

See: architecture_executor.md

4) Answer node

Inputs:

- final_evidence + constraints + answer_requirements

Outputs:

- final user response with citations/format constraints

Design principle:

- Answer node never expands beyond evidence

If coverage is missing blockers, ask clarification instead of guessing

# Contracts and traceability

## State contracts

- Every node reads from and writes to a typed state (TypedDict / Pydantic models)

- Nodes must only write fields they own

- Errors are appended to a shared errors list, never thrown as raw exceptions in normal flow

- Provenance requirements for evidence

Every evidence chunk should preserve:

- doc_id, chunk_id

- retrieval provenance: round_id, query, mode

- rank features: bm25_rank/vector_rank, rrf_score, rerank_score

- metadata: source, title, section, timestamps if available

- Budget controls (hard caps)

At minimum:

- max rounds

- max candidates per round

- max rerank pool

- max final evidence chunks

These are enforced in executor gate and per-round logic.

#Evaluation architecture
## Test categories

Schema contract (hard fail)

- output keys present

- types match

- required fields non-empty

Behavior contract (hard fail for a small reviewed set)

- labeled expectations for a small set of fields

- only assert what you intentionally want stable

Stability (soft fail initially)

- reruns same cases multiple times and checks drift in key fields

# Artifacts

Tests and tools write artifacts to disk for review:

- inputs

- outputs

- expected labels (when present)

- per-round retrieval reports (executor)

# Extension points

Retrieval backend adapters (Azure AI Search, Weaviate, pgvector, etc.)

Cross-encoder reranker model choice and hosting (local, container, Azure)

Policy constraints (sensitivity, ACL filters)

Optional: caching layer for retrieval and rerank results

Non-goals (for now)

Full observability stack (Langfuse, OTEL)

Multi-tenant governance and auditing

Advanced tool use beyond retrieval (browsing, SQL, etc.)





# Intent Intake Subgraph Architecture

The intent (intake) subgraph provides a stable "first look" at the user request. It extracts normalized inputs and planning signals while keeping this step separate from planning and retrieval.

## Purpose

- Standardize the request for logging and downstream planning
- Identify risk and sensitivity signals early
- Detect ambiguity and decide whether clarification is blocking
- Extract structured signals that improve later planning and retrieval

Intake must not:
- retrieve documents
- answer the user
- generate long reasoning text

## Inputs

Primary input:
- `messages`: chat history messages (LangChain message objects or role/content dicts)

Optional inputs:
- `user_context_info`: app-provided context (ACL, org metadata, role)
- `conversation_summary`: optional summary from upstream system

## Outputs

The intake subgraph writes to `IntakeState`:

1) Normalize + gate
- `normalized_query`: cleaned, minimal query for downstream use
- `constraints`: explicit format, prohibitions, domain hints, nonfunctional constraints
- `guardrails`: time sensitivity, context dependency, sensitivity, pii flag
- `clarification`: needed, reasons (and whether blocking if you chose to model that)

2) Signals for planning
- `user_intent`: explain, lookup, compare, decide, troubleshoot, summarize, extract, draft, plan, other
- `retrieval_intent`: none, definition, procedure, evidence, examples, verification, background, mixed
- `answerability`: internal_corpus, external, user_context, reasoning_only, mixed
- `complexity_flags`: multi_intent, multi_domain, requires_synthesis, requires_strict_precision, long_query
- `signals`: entities, acronyms, artifact flags, literal terms

3) Meta
- `intake_version`: version string
- `debug_notes`: short, non-CoT notes (avoid chain-of-thought)

## Node breakdown

The intake subgraph is intentionally small and decomposed:

```mermaid
flowchart LR
  S[START] --> N1[normalize_gate]
  N1 --> N2[extract_signals]
  N2 --> E[END]
