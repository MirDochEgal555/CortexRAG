"""Tests for persistent Confluence vector-store build and search."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.retrieval.vector_store import (
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


class FakeEncoder:
    """Deterministic stand-in for query-time embedding generation."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls: list[dict[str, object]] = []
        self.model_name_or_path = "fake-query-model"

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
        return [self.vectors_by_text[text] for text in texts]


def test_build_confluence_vector_store_and_search_by_embedding(tmp_path: Path) -> None:
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
                "text": "Embeddings convert text into vectors.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "architecture-3309569:002",
                "page": "Architecture",
                "section": "Retrieval",
                "text": "Retrieval finds the nearest chunks.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
            {
                "chunk_id": "architecture-3309569:003",
                "page": "Architecture",
                "section": "Ranking",
                "text": "Ranking compares cosine similarity across vectors.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.8, 0.2],
            },
        ],
    )

    result = build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    assert result.document_count == 3
    assert result.embedding_dimensions == 2
    assert result.embedding_model == "fake-embedding-model"
    assert result.backend in {"chroma", "faiss"}

    manifest = load_vector_store_manifest(
        persist_dir=persist_dir,
        collection_name="test-confluence",
    )
    assert manifest.backend == result.backend
    assert manifest.document_count == 3

    hits = search_confluence_vector_store_by_embedding(
        [1.0, 0.0],
        top_k=2,
        persist_dir=persist_dir,
        collection_name="test-confluence",
    )

    assert [hit.chunk_id for hit in hits] == [
        "architecture-3309569:001",
        "architecture-3309569:003",
    ]
    assert hits[0].score > 0.99
    assert hits[0].metadata["page"] == "Architecture"


def test_query_confluence_vector_store_uses_encoder_and_manifest(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "overview-3178688.jsonl",
        [
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead qualification",
                "text": "The agent qualifies and prioritizes leads.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
            {
                "chunk_id": "overview-3178688:002",
                "page": "Overview",
                "section": "Outreach",
                "text": "The agent drafts outreach for qualified leads.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    encoder = FakeEncoder({"Which chunk is about lead qualification?": [0.0, 1.0]})
    hits = query_confluence_vector_store(
        "Which chunk is about lead qualification?",
        top_k=1,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        encoder=encoder,
    )

    assert [hit.chunk_id for hit in hits] == ["overview-3178688:001"]
    assert hits[0].metadata["section"] == "Lead qualification"
    assert encoder.calls == [
        {
            "texts": ["Which chunk is about lead qualification?"],
            "kwargs": {
                "batch_size": 32,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            },
        }
    ]


def test_similarity_search_confluence_vector_store_supports_min_score(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "overview-3178688.jsonl",
        [
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead qualification",
                "text": "The agent qualifies and prioritizes leads.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
            {
                "chunk_id": "overview-3178688:002",
                "page": "Overview",
                "section": "Outreach",
                "text": "The agent drafts outreach for qualified leads.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.6, 0.8],
            },
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    encoder = FakeEncoder({"How are leads qualified?": [0.0, 1.0]})
    hits = similarity_search_confluence_vector_store(
        "How are leads qualified?",
        top_k=2,
        min_score=0.9,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        encoder=encoder,
    )

    assert [hit.chunk_id for hit in hits] == ["overview-3178688:001"]


def test_similarity_search_by_embedding_supports_min_score(tmp_path: Path) -> None:
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
                "text": "Embeddings convert text into vectors.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "architecture-3309569:002",
                "page": "Architecture",
                "section": "Ranking",
                "text": "Ranking compares cosine similarity across vectors.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            },
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    hits = similarity_search_confluence_vector_store_by_embedding(
        [1.0, 0.0],
        top_k=2,
        min_score=0.95,
        persist_dir=persist_dir,
        collection_name="test-confluence",
    )

    assert [hit.chunk_id for hit in hits] == ["architecture-3309569:001"]


def test_retrieve_confluence_context_reranks_and_deduplicates(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "architecture-3309569.jsonl",
        [
            {
                "chunk_id": "architecture-3309569:001",
                "page": "Architecture",
                "section": "Storage",
                "text": "Storage keeps embeddings persistent and available for retrieval jobs.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.98, 0.02],
            },
            {
                "chunk_id": "architecture-3309569:002",
                "page": "Architecture",
                "section": "Execution Layer",
                "text": (
                    "The execution layer schedules jobs, runs tools, coordinates retries, "
                    "and records outputs for each task."
                ),
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.95, 0.05],
            },
            {
                "chunk_id": "architecture-3309569:003",
                "page": "Architecture",
                "section": "Execution Layer",
                "text": (
                    "The execution layer schedules jobs, runs tools, coordinates retries, "
                    "and records outputs for every task."
                ),
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.94, 0.06],
            },
            {
                "chunk_id": "overview-3178688:001",
                "page": "Delivery Overview",
                "section": "Summary",
                "text": "The overview explains delivery goals, ownership, and rollout scope.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    encoder = FakeEncoder({"How does the execution layer schedule tasks?": [1.0, 0.0]})
    hits = retrieve_confluence_context(
        "How does the execution layer schedule tasks?",
        candidate_k=4,
        final_k=3,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        encoder=encoder,
    )

    assert [hit.chunk_id for hit in hits] == [
        "architecture-3309569:002",
        "architecture-3309569:001",
        "overview-3178688:001",
    ]
    assert hits[0].metadata["retrieval_page_hit_count"] == 3
    assert hits[0].metadata["retrieval_section_keyword_overlap"] == ["execution", "layer"]
    assert hits[0].metadata["retrieval_rerank_score"] > hits[0].metadata["retrieval_similarity_score"]


def test_retrieve_confluence_context_by_embedding_deduplicates_normalized_text(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "overview-3178688.jsonl",
        [
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead Qualification",
                "text": "Qualified leads are scored weekly and routed to sales.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "overview-3178688:002",
                "page": "Overview",
                "section": "Lead Qualification",
                "text": "Qualified leads are scored weekly, and routed to sales.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.99, 0.01],
            },
            {
                "chunk_id": "overview-3178688:003",
                "page": "Overview",
                "section": "Outreach",
                "text": "Qualified leads receive tailored outreach after scoring is complete.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.97, 0.03],
            },
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    hits = retrieve_confluence_context_by_embedding(
        "How are qualified leads routed?",
        [1.0, 0.0],
        candidate_k=3,
        final_k=3,
        persist_dir=persist_dir,
        collection_name="test-confluence",
    )

    assert [hit.chunk_id for hit in hits] == [
        "overview-3178688:001",
        "overview-3178688:003",
    ]


def test_embed_confluence_query_returns_vector_and_manifest(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings" / "confluence" / "ASA"
    persist_dir = tmp_path / "vector-db"
    embeddings_dir.mkdir(parents=True)

    _write_embedding_records(
        embeddings_dir / "overview-3178688.jsonl",
        [
            {
                "chunk_id": "overview-3178688:001",
                "page": "Overview",
                "section": "Lead qualification",
                "text": "The agent qualifies and prioritizes leads.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [0.0, 1.0],
            }
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    encoder = FakeEncoder({"How are leads qualified?": [0.6, 0.8]})
    query_embedding, manifest = embed_confluence_query(
        "How are leads qualified?",
        persist_dir=persist_dir,
        collection_name="test-confluence",
        encoder=encoder,
    )

    assert query_embedding == [0.6, 0.8]
    assert manifest.collection_name == "test-confluence"
    assert manifest.embedding_model == "fake-embedding-model"
    assert encoder.calls == [
        {
            "texts": ["How are leads qualified?"],
            "kwargs": {
                "batch_size": 32,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            },
        }
    ]


def test_search_confluence_vector_store_by_embedding_rejects_wrong_dimensions(tmp_path: Path) -> None:
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
                "text": "Embeddings convert text into vectors.",
                "space_key": "ASA",
                "embedding_model": "fake-embedding-model",
                "embedding_dimensions": 2,
                "embedding": [1.0, 0.0],
            }
        ],
    )

    build_confluence_vector_store(
        input_dir=embeddings_dir.parent,
        persist_dir=persist_dir,
        collection_name="test-confluence",
        backend="auto",
    )

    with pytest.raises(ValueError, match="dimensions do not match"):
        search_confluence_vector_store_by_embedding(
            [1.0, 0.0, 0.0],
            persist_dir=persist_dir,
            collection_name="test-confluence",
        )


def _write_embedding_records(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
