"""Tests for the CortexRAG CLI entry points."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag import cli
from cortex_rag.generation import AnswerTimings, ConfluenceAnswerResult, GenerationResult
from cortex_rag.graph import GraphBuildResult
from cortex_rag.index_status import SearchableIndexStatus
from cortex_rag.retrieval import SearchResult


def test_similarity_search_cli_formats_results(
    monkeypatch,
    capsys,
) -> None:
    def fake_retrieve_context(query: str, **kwargs: object) -> list[SearchResult]:
        assert query == "How are leads qualified?"
        assert kwargs["candidate_k"] == 10
        assert kwargs["final_k"] == 2
        assert kwargs["min_score"] == 0.7
        return [
            SearchResult(
                chunk_id="overview-3178688:001",
                score=0.9321,
                text="The agent qualifies and prioritizes leads.",
                metadata={"page": "Overview", "section": "Lead qualification"},
            )
        ]

    monkeypatch.setattr(cli, "retrieve_confluence_context", fake_retrieve_context)

    cli.main(["similarity-search", "How are leads qualified?", "--top-k", "2", "--min-score", "0.7"])

    assert capsys.readouterr().out.splitlines() == [
        "1. overview-3178688:001  score=0.9321",
        "   Overview :: Lead qualification",
        "   The agent qualifies and prioritizes leads.",
    ]


def test_similarity_search_cli_skips_empty_metadata_line(
    monkeypatch,
    capsys,
) -> None:
    def fake_retrieve_context(query: str, **kwargs: object) -> list[SearchResult]:
        assert query == "What changed?"
        assert kwargs["candidate_k"] == 10
        assert kwargs["final_k"] == 5
        assert kwargs["min_score"] is None
        return [
            SearchResult(
                chunk_id="overview-3178688:002",
                score=0.5,
                text="First line\nSecond line",
                metadata={},
            )
        ]

    monkeypatch.setattr(cli, "retrieve_confluence_context", fake_retrieve_context)

    cli.main(["similarity-search", "What changed?"])

    assert capsys.readouterr().out.splitlines() == [
        "1. overview-3178688:002  score=0.5000",
        "   First line Second line",
    ]


def test_ask_cli_formats_answer_sources_and_timings(
    monkeypatch,
    capsys,
) -> None:
    def fake_answer_question(question: str, **kwargs: object) -> ConfluenceAnswerResult:
        assert question == "What changed?"
        assert kwargs["top_k"] == 2
        assert kwargs["min_score"] == 0.7
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
                generation_seconds=1.5,
                total_seconds=2.25,
                first_token_seconds=0.12,
            ),
        )

    monkeypatch.setattr(cli, "answer_confluence_question", fake_answer_question)

    cli.main(["ask", "What changed?", "--top-k", "2", "--min-score", "0.7"])

    assert capsys.readouterr().out.splitlines() == [
        "Answer:",
        "Grounded answer.",
        "",
        "Sources:",
        "1. overview-3178688:001  score=0.9321",
        "   Overview :: Lead qualification",
        "   The agent qualifies and prioritizes leads.",
        "",
        "Timings:",
        "embedding: 0.50s",
        "retrieval: 0.25s",
        "first_token: 0.12s",
        "generation: 1.50s",
        "total: 2.25s",
    ]


def test_ask_cli_skips_generation_output_when_no_context(
    monkeypatch,
    capsys,
) -> None:
    def fake_answer_question(question: str, **kwargs: object) -> ConfluenceAnswerResult:
        return ConfluenceAnswerResult(
            question=question,
            answer_mode="normal",
            prompt_path=Path("prompts/confluence_rag.md"),
            backend="chroma",
            collection_name="knowledge",
            sources=[],
            messages=[],
            generation=None,
            timings=AnswerTimings(
                embedding_seconds=0.5,
                retrieval_seconds=0.25,
                generation_seconds=0.0,
                total_seconds=1.25,
                first_token_seconds=None,
            ),
        )

    monkeypatch.setattr(cli, "answer_confluence_question", fake_answer_question)

    cli.main(["ask", "What changed?"])

    assert capsys.readouterr().out.splitlines() == [
        "No relevant context was found. Ollama was not called.",
        "",
        "Timings:",
        "embedding: 0.50s",
        "retrieval: 0.25s",
        "generation: 0.00s",
        "total: 1.25s",
    ]


def test_build_graph_cli_prints_summary(
    monkeypatch,
    capsys,
) -> None:
    def fake_build_graph(**kwargs: object) -> GraphBuildResult:
        assert kwargs["collection_name"] == "knowledge"
        assert kwargs["similarity_top_k"] == 2
        assert kwargs["similarity_threshold"] == 0.7
        return GraphBuildResult(
            collection_name="knowledge",
            persist_path=Path("storage/chroma/knowledge.graph.json"),
            document_node_count=2,
            chunk_node_count=3,
            belongs_to_edge_count=3,
            similar_to_edge_count=1,
            similarity_top_k=2,
            similarity_threshold=0.7,
        )

    monkeypatch.setattr(cli, "build_knowledge_graph", fake_build_graph)

    cli.main(["build-graph", "--similarity-top-k", "2", "--similarity-threshold", "0.7"])

    assert capsys.readouterr().out.splitlines() == [
        "Built graph 'knowledge' with 5 nodes and 4 edges at storage\\chroma\\knowledge.graph.json.",
    ]


def test_build_vector_store_cli_can_also_build_graph(
    monkeypatch,
    capsys,
) -> None:
    class FakeVectorStoreBuildResult:
        backend = "chroma"
        collection_name = "knowledge"
        document_count = 3
        persist_dir = Path("storage/chroma")

    def fake_build_vector_store(**kwargs: object) -> FakeVectorStoreBuildResult:
        assert kwargs["collection_name"] == "knowledge"
        return FakeVectorStoreBuildResult()

    def fake_build_graph(**kwargs: object) -> GraphBuildResult:
        assert kwargs["collection_name"] == "knowledge"
        assert kwargs["similarity_top_k"] == 4
        assert kwargs["similarity_threshold"] == 0.8
        return GraphBuildResult(
            collection_name="knowledge",
            persist_path=Path("storage/chroma/knowledge.graph.json"),
            document_node_count=2,
            chunk_node_count=3,
            belongs_to_edge_count=3,
            similar_to_edge_count=2,
            similarity_top_k=4,
            similarity_threshold=0.8,
        )

    monkeypatch.setattr(cli, "build_knowledge_vector_store", fake_build_vector_store)
    monkeypatch.setattr(cli, "build_knowledge_graph", fake_build_graph)

    cli.main(
        [
            "build-vector-store",
            "--with-graph",
            "--graph-similarity-top-k",
            "4",
            "--graph-similarity-threshold",
            "0.8",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "Built chroma vector store 'knowledge' with 3 chunks at storage\\chroma.",
        "Built graph 'knowledge' with 5 nodes and 5 edges at storage\\chroma\\knowledge.graph.json.",
    ]


def test_verify_index_cli_prints_current_searchable_index(
    monkeypatch,
    capsys,
) -> None:
    def fake_verify_index(**kwargs: object) -> SearchableIndexStatus:
        assert kwargs["collection_name"] == "knowledge"
        assert kwargs["backend"] == "auto"
        return SearchableIndexStatus(
            collection_name="knowledge",
            persist_dir=Path("storage/chroma"),
            backend="faiss",
            document_count=3,
            embedding_model="fake-embedding-model",
            graph_node_count=5,
            graph_edge_count=4,
            manifest_path=Path("storage/chroma/knowledge.manifest.json"),
            graph_path=Path("storage/chroma/knowledge.graph.json"),
        )

    monkeypatch.setattr(cli, "verify_current_searchable_index", fake_verify_index)

    cli.main(["verify-index"])

    assert capsys.readouterr().out.splitlines() == [
        f"Verified faiss searchable index 'knowledge' with 3 chunks at {Path('storage/chroma')}.",
        "Embedding model: fake-embedding-model",
        f"Manifest: {Path('storage/chroma/knowledge.manifest.json')}",
        f"Graph: {Path('storage/chroma/knowledge.graph.json')}",
        "Graph payload: 5 nodes, 4 edges.",
    ]
