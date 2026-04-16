"""Embedding and vector-store retrieval components."""

from cortex_rag.retrieval.confluence_embeddings import (
    generate_confluence_embeddings,
    generate_confluence_space_embeddings,
)
from cortex_rag.retrieval.vector_store import (
    SearchResult,
    VectorStoreBuildResult,
    build_confluence_vector_store,
    embed_confluence_query,
    load_vector_store_manifest,
    query_confluence_vector_store,
    retrieve_confluence_context,
    retrieve_confluence_context_by_embedding,
    similarity_search_confluence_vector_store,
    similarity_search_confluence_vector_store_by_embedding,
    search_confluence_vector_store_by_embedding,
)

__all__ = [
    "SearchResult",
    "VectorStoreBuildResult",
    "build_confluence_vector_store",
    "embed_confluence_query",
    "generate_confluence_embeddings",
    "generate_confluence_space_embeddings",
    "load_vector_store_manifest",
    "query_confluence_vector_store",
    "retrieve_confluence_context",
    "retrieve_confluence_context_by_embedding",
    "similarity_search_confluence_vector_store",
    "similarity_search_confluence_vector_store_by_embedding",
    "search_confluence_vector_store_by_embedding",
]
