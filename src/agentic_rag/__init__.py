"""Agentic RAG system using LangGraph.

This package provides an agentic RAG pipeline with clear separation between:
- Intent intake: normalize and extract planning signals
- Planner: produce executable search plans
- Executor: run retrieval, merge, rerank, and select evidence
- Answer: synthesize final response from evidence
"""

__version__ = "0.1.0"
