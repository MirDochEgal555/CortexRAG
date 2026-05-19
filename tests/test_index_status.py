"""Tests for searchable index preflight checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag import index_status
from cortex_rag.retrieval import VectorStoreArtifactStatus


def test_verify_current_searchable_index_requires_matching_graph(monkeypatch, tmp_path: Path) -> None:
    def fake_verify_vector_store_artifacts(**kwargs: object) -> VectorStoreArtifactStatus:
        return VectorStoreArtifactStatus(
            backend="faiss",
            collection_name="knowledge",
            persist_dir=tmp_path,
            document_count=3,
            embedding_model="fake-embedding-model",
            manifest_path=tmp_path / "knowledge.manifest.json",
        )

    def fake_load_graph(**kwargs: object) -> object:
        return SimpleNamespace(chunk_node_count=3, nodes=[object()] * 5, edges=[object()] * 4)

    monkeypatch.setattr(index_status, "verify_vector_store_artifacts", fake_verify_vector_store_artifacts)
    monkeypatch.setattr(index_status, "load_confluence_graph", fake_load_graph)

    status = index_status.verify_current_searchable_index(persist_dir=tmp_path)

    assert status.collection_name == "knowledge"
    assert status.backend == "faiss"
    assert status.document_count == 3
    assert status.embedding_model == "fake-embedding-model"
    assert status.graph_node_count == 5
    assert status.graph_edge_count == 4
    assert status.manifest_path == tmp_path / "knowledge.manifest.json"
    assert status.graph_path == tmp_path / "knowledge.graph.json"


def test_verify_current_searchable_index_rejects_out_of_sync_graph(monkeypatch, tmp_path: Path) -> None:
    def fake_verify_vector_store_artifacts(**kwargs: object) -> VectorStoreArtifactStatus:
        return VectorStoreArtifactStatus(
            backend="faiss",
            collection_name="knowledge",
            persist_dir=tmp_path,
            document_count=3,
            embedding_model="fake-embedding-model",
            manifest_path=tmp_path / "knowledge.manifest.json",
        )

    def fake_load_graph(**kwargs: object) -> object:
        return SimpleNamespace(chunk_node_count=2, nodes=[], edges=[])

    monkeypatch.setattr(index_status, "verify_vector_store_artifacts", fake_verify_vector_store_artifacts)
    monkeypatch.setattr(index_status, "load_confluence_graph", fake_load_graph)

    with pytest.raises(ValueError, match="Graph artifact is out of sync"):
        index_status.verify_current_searchable_index(persist_dir=tmp_path)
