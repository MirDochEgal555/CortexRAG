"""Tests for the persisted Confluence graph artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.graph import build_confluence_graph, build_graph_neighborhood, load_confluence_graph


def test_build_confluence_graph_persists_document_chunk_and_similarity_edges(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "architecture-3309569.jsonl",
        [
            {
                "chunk_id": "architecture-3309569:001",
                "page": "Architecture",
                "section": "Embeddings",
                "headings": ["Architecture", "Embeddings"],
                "text": "Embeddings convert text into vectors.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/architecture-3309569.md",
                "word_count": 210,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "architecture-3309569:002",
                "page": "Architecture",
                "section": "Retrieval",
                "headings": ["Architecture", "Retrieval"],
                "text": "Retrieval finds the nearest chunks.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/architecture-3309569.md",
                "word_count": 230,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.9, 0.1],
            },
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead qualification",
                "headings": ["Overview", "Lead qualification"],
                "text": "The agent qualifies and prioritizes leads.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/overview-3178688.md",
                "word_count": 220,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
        ],
    )

    result = build_confluence_graph(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        similarity_top_k=1,
        similarity_threshold=0.7,
    )

    assert result.collection_name == "test-confluence"
    assert result.document_node_count == 2
    assert result.chunk_node_count == 3
    assert result.belongs_to_edge_count == 3
    assert result.similar_to_edge_count == 1
    assert result.persist_path == persist_dir / "test-confluence.graph.json"
    assert result.persist_path.exists()

    graph = load_confluence_graph(persist_dir=persist_dir, collection_name="test-confluence")

    assert graph.document_node_count == 2
    assert graph.chunk_node_count == 3
    assert {node.id for node in graph.nodes} == {
        "document::ASA/architecture-3309569.md",
        "document::ASA/overview-3178688.md",
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
        "chunk::overview-3178688:001",
    }
    assert {edge.type for edge in graph.edges} == {"belongs_to", "similar_to"}
    assert any(edge.id == "chunk::architecture-3309569:001--chunk::architecture-3309569:002::similar_to" for edge in graph.edges)
    document_node = next(node for node in graph.nodes if node.id == "document::ASA/architecture-3309569.md")
    chunk_node = next(node for node in graph.nodes if node.id == "chunk::architecture-3309569:001")
    belongs_to_edge = next(
        edge for edge in graph.edges if edge.id == "document::ASA/architecture-3309569.md--chunk::architecture-3309569:001::belongs_to"
    )
    similarity_edge = next(
        edge for edge in graph.edges if edge.id == "chunk::architecture-3309569:001--chunk::architecture-3309569:002::similar_to"
    )
    assert document_node.metadata["summary"] == "Embeddings convert text into vectors."
    assert chunk_node.metadata["summary"] == "Embeddings convert text into vectors."
    assert belongs_to_edge.metadata["reason"] == "same_document"
    assert similarity_edge.metadata["reason"] == "nearest_neighbor_similarity"
    assert similarity_edge.metadata["shared_metadata"] == ["same source (confluence)", "same space (ASA)"]


def test_build_confluence_graph_prefers_document_id_for_source_neutral_chunks(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "knowledge" / "zotero"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "paper.jsonl",
        [
            {
                "chunk_id": "zotero::doe2024rag:001",
                "document_id": "zotero::doe2024rag",
                "page": "RAG for Knowledge Work",
                "section": "Notes",
                "text": "Zotero annotation text.",
                "source": "zotero",
                "source_path": "library.bib",
                "word_count": 12,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "zotero::doe2024rag:002",
                "document_id": "zotero::doe2024rag",
                "page": "RAG for Knowledge Work",
                "section": "Bibliography",
                "text": "Bibliographic metadata text.",
                "source": "zotero",
                "source_path": "library.bib",
                "word_count": 8,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.9, 0.1],
            },
        ],
    )

    build_confluence_graph(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="knowledge",
        similarity_top_k=1,
        similarity_threshold=0.7,
    )

    graph = load_confluence_graph(persist_dir=persist_dir, collection_name="knowledge")

    assert graph.document_node_count == 1
    assert any(node.id == "document::zotero::doe2024rag" for node in graph.nodes)


def test_build_graph_neighborhood_expands_seed_chunks_to_documents_and_similar_neighbors(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "architecture-3309569.jsonl",
        [
            {
                "chunk_id": "architecture-3309569:001",
                "page": "Architecture",
                "section": "Embeddings",
                "headings": ["Architecture", "Embeddings"],
                "text": "Embeddings convert text into vectors.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/architecture-3309569.md",
                "word_count": 210,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "architecture-3309569:002",
                "page": "Architecture",
                "section": "Retrieval",
                "headings": ["Architecture", "Retrieval"],
                "text": "Retrieval finds the nearest chunks.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/architecture-3309569.md",
                "word_count": 230,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.9, 0.1],
            },
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead qualification",
                "headings": ["Overview", "Lead qualification"],
                "text": "The agent qualifies and prioritizes leads.",
                "source": "confluence",
                "space_key": "ASA",
                "source_path": "ASA/overview-3178688.md",
                "word_count": 220,
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
        ],
    )

    build_confluence_graph(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        similarity_top_k=1,
        similarity_threshold=0.7,
    )
    graph = load_confluence_graph(persist_dir=persist_dir, collection_name="test-confluence")

    neighborhood = build_graph_neighborhood(
        graph,
        seed_chunk_ids=["architecture-3309569:001"],
    )

    assert neighborhood.seed_node_ids == ["chunk::architecture-3309569:001"]
    assert set(neighborhood.highlighted_node_ids) == {
        "chunk::architecture-3309569:001",
        "document::ASA/architecture-3309569.md",
    }
    assert set(neighborhood.query_path_node_ids) == {
        "document::ASA/architecture-3309569.md",
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
    }
    assert {node.id for node in neighborhood.nodes} == {
        "document::ASA/architecture-3309569.md",
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
    }
    assert {edge.id for edge in neighborhood.edges} == {
        "document::ASA/architecture-3309569.md--chunk::architecture-3309569:001::belongs_to",
        "document::ASA/architecture-3309569.md--chunk::architecture-3309569:002::belongs_to",
        "chunk::architecture-3309569:001--chunk::architecture-3309569:002::similar_to",
    }
    assert set(neighborhood.query_path_edge_ids) == {
        "document::ASA/architecture-3309569.md--chunk::architecture-3309569:001::belongs_to",
        "document::ASA/architecture-3309569.md--chunk::architecture-3309569:002::belongs_to",
        "chunk::architecture-3309569:001--chunk::architecture-3309569:002::similar_to",
    }


def _write_embedding_records(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
