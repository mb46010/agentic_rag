# src/agentic_rag/executor/constants.py
"""Constants for executor configuration.

These defaults can be overridden via plan specifications.
"""

# RRF (Reciprocal Rank Fusion) parameters
DEFAULT_RRF_K = 60  # RRF smoothing constant
DEFAULT_RRF_POOL_SIZE = 200  # Max candidates to consider for fusion

# Evidence selection limits
DEFAULT_MAX_TOTAL_DOCS = 12  # Max documents in final evidence pack
DEFAULT_MAX_DOCS_PER_ROUND = 8  # Max documents selected per round

# Reranking parameters
DEFAULT_RERANK_TOP_K = 60  # Max candidates to send to reranker

# Retrieval parameters
DEFAULT_RETRIEVAL_K = 20  # Default number of candidates per retrieval call
