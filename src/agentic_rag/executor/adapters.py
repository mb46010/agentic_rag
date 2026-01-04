# src/agentic_rag/executor/adapters.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

from agentic_rag.executor.state import Candidate


class RetrieverAdapter(Protocol):
    """Adapter for retrieval backends (Azure AI Search, Weaviate, pgvector, etc.)."""

    def search(
        self,
        *,
        query: str,
        mode: str,  # "bm25" | "vector" | "hybrid"
        k: int,
        alpha: Optional[float],
        filters: Dict[str, Any],
    ) -> List[Candidate]:
        """Return candidates. Must populate Candidate.key, text, metadata, and retrieval features as available."""
        raise NotImplementedError


class HyDEAdapter(Protocol):
    """Adapter for HyDE generation, can use an LLM or a deterministic template."""

    def synthesize(self, *, query: str, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    def derive_queries(self, *, original_query: str, synthetic_answer: str, max_queries: int) -> List[str]:
        raise NotImplementedError


class RerankerAdapter(Protocol):
    """Adapter for cross-encoder reranking."""

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        context: Dict[str, Any],
    ) -> List[Candidate]:
        """Return candidates sorted by relevance and with rerank_score set."""
        raise NotImplementedError


class FusionAdapter(Protocol):
    """Adapter for fusion like RRF."""

    def rrf(self, *, ranked_lists: List[List[Candidate]], k: int = 60, rrf_k: int = 60) -> List[Candidate]:
        raise NotImplementedError


class CoverageGraderAdapter(Protocol):
    """Optional LLM-based grader. Keep it structured and bounded."""

    def grade(
        self,
        *,
        plan: Dict[str, Any],
        normalized_query: str,
        selected_evidence: Sequence[Candidate],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Returns Coverage dict-like object."""
        raise NotImplementedError


# -------------------------
# Simple defaults (placeholders)
# -------------------------


class NotImplementedRetriever:
    def search(
        self, *, query: str, mode: str, k: int, alpha: Optional[float], filters: Dict[str, Any]
    ) -> List[Candidate]:
        raise NotImplementedError("Provide a RetrieverAdapter implementation")


class NotImplementedHyDE:
    def synthesize(self, *, query: str, context: Dict[str, Any]) -> str:
        raise NotImplementedError("Provide a HyDEAdapter implementation")

    def derive_queries(self, *, original_query: str, synthetic_answer: str, max_queries: int) -> List[str]:
        raise NotImplementedError("Provide a HyDEAdapter implementation")


class NotImplementedReranker:
    def rerank(
        self, *, query: str, candidates: Sequence[Candidate], top_k: int, context: Dict[str, Any]
    ) -> List[Candidate]:
        raise NotImplementedError("Provide a RerankerAdapter implementation")


class SimpleRRF:
    """Deterministic RRF fusion. Safe default until you swap an adapter."""

    def rrf(self, *, ranked_lists: List[List[Candidate]], k: int = 60, rrf_k: int = 60) -> List[Candidate]:
        # Simple reciprocal rank fusion across multiple ranked lists.
        # Candidate identity is Candidate.key.
        scores: Dict[Any, float] = {}
        best: Dict[Any, Candidate] = {}

        for lst in ranked_lists:
            for rank, cand in enumerate(lst, start=1):
                key = cand.key
                best[key] = cand
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        fused = list(best.values())
        for cand in fused:
            cand.rrf_score = scores.get(cand.key, 0.0)

        fused.sort(key=lambda c: (c.rrf_score or 0.0), reverse=True)
        return fused[:k]


class NoOpCoverageGrader:
    def grade(
        self,
        *,
        plan: Dict[str, Any],
        normalized_query: str,
        selected_evidence: Sequence[Candidate],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Conservative: do not claim coverage, force stop by max_rounds.
        return {
            "covered_entities": [],
            "missing_entities": (plan.get("acceptance_criteria") or {}).get("must_cover_entities", []) or [],
            "covered_subquestions": [],
            "missing_subquestions": (plan.get("acceptance_criteria") or {}).get("must_answer_subquestions", []) or [],
            "evidence_quality": "low",
            "confidence": 0.0,
            "contradictions": [],
        }
