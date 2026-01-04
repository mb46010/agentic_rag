# src/agentic_rag/executor/adapters.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

from agentic_rag.executor.state import Candidate


class RetrieverAdapter(Protocol):
    """Adapter for retrieval backends (Azure AI Search, Weaviate, pgvector, etc.).

    Example implementation for Azure AI Search:

        from azure.search.documents import SearchClient
        from agentic_rag.executor.state import Candidate, CandidateKey

        class AzureSearchRetriever:
            def __init__(self, search_client: SearchClient):
                self.client = search_client

            def search(self, *, query, mode, k, alpha, filters):
                # Map mode to Azure search parameters
                if mode == "bm25":
                    results = self.client.search(query, top=k, query_type="simple")
                elif mode == "vector":
                    results = self.client.search(query, top=k, query_type="semantic")
                elif mode == "hybrid":
                    results = self.client.search(
                        query, top=k, query_type="semantic",
                        hybrid_parameters={"weight": alpha or 0.5}
                    )

                # Map to Candidate objects
                candidates = []
                for rank, hit in enumerate(results):
                    candidates.append(Candidate(
                        key=CandidateKey(doc_id=hit["id"], chunk_id=hit["chunk_id"]),
                        text=hit["content"],
                        metadata={"source": hit.get("source"), "title": hit.get("title")},
                        bm25_score=hit.get("@search.score"),
                        bm25_rank=rank,
                    ))
                return candidates
    """

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
    """Adapter for HyDE (Hypothetical Document Embeddings) generation.

    HyDE generates synthetic answers to expand query variants for better retrieval.

    Example implementation:

        class LLMHyDEAdapter:
            def __init__(self, llm):
                self.llm = llm

            def synthesize(self, *, query, context):
                prompt = f"Write a short, direct answer to: {query}"
                return self.llm.invoke(prompt).content

            def derive_queries(self, *, original_query, synthetic_answer, max_queries):
                prompt = f'''Given this query: "{original_query}"
                And this hypothetical answer: "{synthetic_answer}"
                Generate {max_queries} alternative search queries.'''
                result = self.llm.invoke(prompt).content
                return result.split("\\n")[:max_queries]
    """

    def synthesize(self, *, query: str, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    def derive_queries(self, *, original_query: str, synthetic_answer: str, max_queries: int) -> List[str]:
        raise NotImplementedError


class RerankerAdapter(Protocol):
    """Adapter for cross-encoder reranking.

    Example implementation using a cross-encoder model:

        from sentence_transformers import CrossEncoder
        from dataclasses import replace

        class CrossEncoderReranker:
            def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
                self.model = CrossEncoder(model_name)

            def rerank(self, *, query, candidates, top_k, context):
                if not candidates:
                    return []

                # Prepare query-document pairs
                pairs = [(query, c.text) for c in candidates]

                # Score all pairs
                scores = self.model.predict(pairs)

                # Attach scores and sort
                scored = []
                for cand, score in zip(candidates, scores):
                    scored.append(replace(cand, rerank_score=float(score)))

                scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
                return scored[:top_k]
    """

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
    """Optional LLM-based grader for evaluating evidence coverage.

    Example implementation:

        class LLMCoverageGrader:
            def __init__(self, llm):
                self.llm = llm.with_structured_output(CoverageModel)

            def grade(self, *, plan, normalized_query, selected_evidence, context):
                acceptance = plan.get("acceptance_criteria", {})
                must_cover = acceptance.get("must_cover_entities", [])

                evidence_text = "\\n".join([c.text[:200] for c in selected_evidence])

                prompt = f'''Query: {normalized_query}
                Required entities: {must_cover}
                Evidence: {evidence_text}

                Grade coverage:
                - Which entities are covered?
                - What's missing?
                - Evidence quality (high/medium/low)?
                - Confidence (0-1)?
                '''

                result = self.llm.invoke(prompt)
                return result.model_dump()
    """

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
