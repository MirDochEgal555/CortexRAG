"""Runtime checks for the searchable index used by the UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cortex_rag.config import DEFAULT_VECTOR_COLLECTION, VECTOR_DB_DIR
from cortex_rag.graph import load_confluence_graph
from cortex_rag.retrieval import VectorStoreArtifactStatus, VectorBackend, verify_vector_store_artifacts


@dataclass(frozen=True)
class SearchableIndexStatus:
    """Verified state of the current UI-searchable index."""

    collection_name: str
    persist_dir: Path
    backend: str
    document_count: int
    embedding_model: str
    graph_node_count: int
    graph_edge_count: int
    manifest_path: Path
    graph_path: Path


def verify_current_searchable_index(
    *,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
) -> SearchableIndexStatus:
    """Verify that the Zotero/Obsidian knowledge vector store and graph are queryable by the UI."""

    vector_status = verify_vector_store_artifacts(
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
    )
    graph = load_confluence_graph(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    if graph.chunk_node_count != vector_status.document_count:
        raise ValueError(
            "Graph artifact is out of sync with the vector store manifest: "
            f"{graph.chunk_node_count} graph chunks != {vector_status.document_count} vector documents."
        )

    return _build_status(vector_status, graph_node_count=len(graph.nodes), graph_edge_count=len(graph.edges))


def _build_status(
    vector_status: VectorStoreArtifactStatus,
    *,
    graph_node_count: int,
    graph_edge_count: int,
) -> SearchableIndexStatus:
    graph_path = vector_status.persist_dir / f"{vector_status.collection_name}.graph.json"
    return SearchableIndexStatus(
        collection_name=vector_status.collection_name,
        persist_dir=vector_status.persist_dir,
        backend=vector_status.backend,
        document_count=vector_status.document_count,
        embedding_model=vector_status.embedding_model,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        manifest_path=vector_status.manifest_path,
        graph_path=graph_path,
    )
