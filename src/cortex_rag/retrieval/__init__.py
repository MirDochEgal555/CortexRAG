"""Embedding and vector-store retrieval components."""

from cortex_rag.retrieval.confluence_embeddings import (
    generate_confluence_embeddings,
    generate_confluence_space_embeddings,
)

__all__ = [
    "generate_confluence_embeddings",
    "generate_confluence_space_embeddings",
]
