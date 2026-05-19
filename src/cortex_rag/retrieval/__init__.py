"""Embedding and vector-store retrieval components."""

from cortex_rag.retrieval.confluence_embeddings import (
    generate_confluence_embeddings,
    generate_confluence_space_embeddings,
)
from cortex_rag.retrieval.chunk_embeddings import (
    generate_chunk_embeddings,
    generate_knowledge_embeddings,
)
from cortex_rag.retrieval.embedding_utils import (
    clear_sentence_transformer_cache,
    preload_sentence_transformer,
)
from cortex_rag.retrieval.vector_store import (
    SearchResult,
    VectorBackend,
    VectorStoreArtifactStatus,
    VectorStoreBuildResult,
    build_confluence_vector_store,
    build_knowledge_vector_store,
    embed_confluence_query,
    load_vector_store_manifest,
    query_confluence_vector_store,
    retrieve_confluence_context,
    retrieve_confluence_context_by_embedding,
    similarity_search_confluence_vector_store,
    similarity_search_confluence_vector_store_by_embedding,
    search_confluence_vector_store_by_embedding,
    verify_vector_store_artifacts,
)

__all__ = [
    "SearchResult",
    "VectorBackend",
    "VectorStoreArtifactStatus",
    "VectorStoreBuildResult",
    "build_confluence_vector_store",
    "build_knowledge_vector_store",
    "clear_sentence_transformer_cache",
    "embed_confluence_query",
    "generate_chunk_embeddings",
    "generate_confluence_embeddings",
    "generate_confluence_space_embeddings",
    "generate_knowledge_embeddings",
    "load_vector_store_manifest",
    "preload_sentence_transformer",
    "query_confluence_vector_store",
    "retrieve_confluence_context",
    "retrieve_confluence_context_by_embedding",
    "similarity_search_confluence_vector_store",
    "similarity_search_confluence_vector_store_by_embedding",
    "search_confluence_vector_store_by_embedding",
    "verify_vector_store_artifacts",
]
