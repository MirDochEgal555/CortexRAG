"""Tests for the FastAPI app when optional API dependencies are installed."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        return False
    return True


def _has_httpx() -> bool:
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def test_create_app_raises_clear_error_without_fastapi(monkeypatch) -> None:
    if _has_fastapi():
        pytest.skip("FastAPI is installed in this environment.")

    from cortex_rag.api.app import create_app

    with pytest.raises(RuntimeError, match="FastAPI is not installed"):
        create_app()


@pytest.mark.skipif(not (_has_fastapi() and _has_httpx()), reason="FastAPI test dependencies are not installed.")
def test_health_endpoint_returns_ok(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from cortex_rag.api import app as api_app

    warmed: list[tuple[str, object]] = []

    def fake_warm_ui_runtime_assets(*, persist_dir=None, collection_name="knowledge") -> None:
        warmed.append((collection_name, persist_dir))

    monkeypatch.setattr(api_app, "_WARMED_UI_RUNTIME_KEYS", set())
    monkeypatch.setattr(api_app, "warm_ui_runtime_assets", fake_warm_ui_runtime_assets)
    client = TestClient(api_app.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "cortex_rag"}
    assert len(warmed) == 1
    assert warmed[0][0] == "knowledge"


@pytest.mark.skipif(not (_has_fastapi() and _has_httpx()), reason="FastAPI test dependencies are not installed.")
def test_search_endpoint_serializes_results(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from cortex_rag.api import app as api_app
    from cortex_rag.retrieval import SearchFilters, SearchResult

    def fake_retrieve_context(query: str, **kwargs: object) -> list[SearchResult]:
        assert query == "What changed?"
        assert kwargs["final_k"] == 3
        assert kwargs["filters"] == SearchFilters(source="zotero", citekey="doe2024rag")
        return [
            SearchResult(
                chunk_id="overview-3178688:001",
                score=0.9321,
                text="The agent qualifies and prioritizes leads.",
                metadata={"page": "Overview", "section": "Lead qualification"},
            )
        ]

    warmed: list[tuple[str, object]] = []

    def fake_warm_ui_runtime_assets(*, persist_dir=None, collection_name="knowledge") -> None:
        warmed.append((collection_name, persist_dir))

    monkeypatch.setattr(api_app, "_WARMED_UI_RUNTIME_KEYS", set())
    monkeypatch.setattr(api_app, "warm_ui_runtime_assets", fake_warm_ui_runtime_assets)
    monkeypatch.setattr(api_app, "retrieve_confluence_context", fake_retrieve_context)
    client = TestClient(api_app.create_app())

    response = client.post(
        "/search",
        json={"query": "What changed?", "top_k": 3, "source": "zotero", "citekey": "doe2024rag"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What changed?",
        "result_count": 1,
        "results": [
            {
                "chunk_id": "overview-3178688:001",
                "score": 0.9321,
                "text": "The agent qualifies and prioritizes leads.",
                "metadata": {"page": "Overview", "section": "Lead qualification"},
            }
        ],
    }
    assert len(warmed) == 1
    assert warmed[0][0] == "knowledge"


@pytest.mark.skipif(not (_has_fastapi() and _has_httpx()), reason="FastAPI test dependencies are not installed.")
def test_answer_endpoint_serializes_grounded_answer(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from cortex_rag.api import app as api_app
    from cortex_rag.generation import AnswerTimings, ConfluenceAnswerResult, GenerationResult
    from cortex_rag.retrieval import SearchFilters, SearchResult

    def fake_answer_question(question: str, **kwargs: object) -> ConfluenceAnswerResult:
        assert question == "What changed?"
        assert kwargs["top_k"] == 2
        assert kwargs["filters"] == SearchFilters(source="zotero", citekey="doe2024rag")
        return ConfluenceAnswerResult(
            question=question,
            answer_mode="normal",
            prompt_path=Path("prompts/confluence_rag.md"),
            backend="chroma",
            collection_name="knowledge",
            sources=[
                SearchResult(
                    chunk_id="overview-3178688:001",
                    score=0.9321,
                    text="The agent qualifies and prioritizes leads.",
                    metadata={"page": "Overview", "section": "Lead qualification"},
                )
            ],
            messages=[{"role": "user", "content": "Question:\nWhat changed?"}],
            generation=GenerationResult(
                model="llama3.2:3b",
                content="Grounded answer.",
                first_token_seconds=0.12,
                prompt_eval_count=42,
                eval_count=11,
                done_reason="stop",
            ),
            timings=AnswerTimings(
                embedding_seconds=0.5,
                retrieval_seconds=0.25,
                generation_seconds=1.25,
                total_seconds=2.0,
                first_token_seconds=0.12,
            ),
        )

    monkeypatch.setattr(api_app, "answer_confluence_question", fake_answer_question)
    client = TestClient(api_app.create_app())

    response = client.post("/answer", json={"query": "What changed?", "source": "zotero", "citekey": "doe2024rag"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer."
    assert response.json()["generated"] is True
    assert response.json()["backend"] == "chroma"
    assert response.json()["sources"][0]["chunk_id"] == "overview-3178688:001"


@pytest.mark.skipif(not (_has_fastapi() and _has_httpx()), reason="FastAPI test dependencies are not installed.")
def test_graph_neighborhood_endpoint_returns_nodes_and_edges(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from cortex_rag.api import app as api_app
    from cortex_rag.graph import GraphArtifact, GraphEdge, GraphNode
    from cortex_rag.retrieval import SearchFilters, SearchResult

    def fake_retrieve_context(query: str, **kwargs: object) -> list[SearchResult]:
        assert query == "What changed?"
        assert kwargs["filters"] == SearchFilters(source="zotero", citekey="doe2024rag")
        return [
            SearchResult(
                chunk_id="architecture-3309569:001",
                score=0.91,
                text="The execution layer runs retrieval and generation steps.",
                metadata={
                    "page": "Architecture",
                    "section": "Execution layer",
                    "source_path": "ASA/architecture-3309569.md",
                },
            ),
            SearchResult(
                chunk_id="architecture-3309569:002",
                score=0.87,
                text="The orchestration layer schedules retrieval tasks.",
                metadata={
                    "page": "Architecture",
                    "section": "Orchestration",
                    "source_path": "ASA/architecture-3309569.md",
                },
            ),
        ]

    def fake_load_graph(*, persist_dir, collection_name):
        assert collection_name == "knowledge"
        return GraphArtifact(
            collection_name="knowledge",
            document_node_count=1,
            chunk_node_count=2,
            belongs_to_edge_count=2,
            similar_to_edge_count=1,
            similarity_top_k=3,
            similarity_threshold=0.6,
            nodes=[
                GraphNode(
                    id="document::ASA/architecture-3309569.md",
                    type="document",
                    label="Architecture",
                    metadata={"page": "Architecture"},
                ),
                GraphNode(
                    id="chunk::architecture-3309569:001",
                    type="chunk",
                    label="Execution layer",
                    metadata={"chunk_id": "architecture-3309569:001"},
                ),
                GraphNode(
                    id="chunk::architecture-3309569:002",
                    type="chunk",
                    label="Orchestration",
                    metadata={"chunk_id": "architecture-3309569:002"},
                ),
            ],
            edges=[
                GraphEdge(
                    id="document::ASA/architecture-3309569.md--chunk::architecture-3309569:001::belongs_to",
                    source="document::ASA/architecture-3309569.md",
                    target="chunk::architecture-3309569:001",
                    type="belongs_to",
                    weight=1.0,
                    metadata={"reason": "chunk_source_membership"},
                ),
                GraphEdge(
                    id="document::ASA/architecture-3309569.md--chunk::architecture-3309569:002::belongs_to",
                    source="document::ASA/architecture-3309569.md",
                    target="chunk::architecture-3309569:002",
                    type="belongs_to",
                    weight=1.0,
                    metadata={"reason": "chunk_source_membership"},
                ),
                GraphEdge(
                    id="chunk::architecture-3309569:001--chunk::architecture-3309569:002::similar_to",
                    source="chunk::architecture-3309569:001",
                    target="chunk::architecture-3309569:002",
                    type="similar_to",
                    weight=0.88,
                    metadata={"reason": "embedding_similarity"},
                ),
            ],
        )

    monkeypatch.setattr(api_app, "retrieve_confluence_context", fake_retrieve_context)
    monkeypatch.setattr(api_app, "load_confluence_graph", fake_load_graph)
    client = TestClient(api_app.create_app())

    response = client.post(
        "/graph/neighborhood",
        json={"query": "What changed?", "source": "zotero", "citekey": "doe2024rag"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_count"] == 2
    assert payload["seed_node_ids"] == [
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
    ]
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 3
