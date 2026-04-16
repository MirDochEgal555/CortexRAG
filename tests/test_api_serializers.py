"""Tests for API serialization helpers."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.api.serializers import (
    build_answer_response,
    build_graph_neighborhood_response,
    build_search_response,
)
from cortex_rag.generation import AnswerTimings, ConfluenceAnswerResult, GenerationResult
from cortex_rag.retrieval import SearchResult


def test_build_search_response_serializes_results() -> None:
    response = build_search_response(
        "What changed?",
        [
            SearchResult(
                chunk_id="overview-3178688:001",
                score=0.9321,
                text="The agent qualifies and prioritizes leads.",
                metadata={"page": "Overview", "section": "Lead qualification"},
            )
        ],
    )

    assert response.query == "What changed?"
    assert response.result_count == 1
    assert response.results[0].chunk_id == "overview-3178688:001"
    assert response.results[0].metadata["page"] == "Overview"


def test_build_answer_response_serializes_structured_answer() -> None:
    result = ConfluenceAnswerResult(
        question="What changed?",
        answer_mode="technical",
        prompt_path=Path("prompts/confluence_rag.md"),
        backend="chroma",
        collection_name="confluence",
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
            first_token_seconds=0.14,
            prompt_eval_count=42,
            eval_count=11,
            done_reason="stop",
        ),
        timings=AnswerTimings(
            embedding_seconds=0.5,
            retrieval_seconds=0.25,
            generation_seconds=1.0,
            total_seconds=1.75,
            first_token_seconds=0.14,
        ),
    )

    response = build_answer_response(result)

    assert response.question == "What changed?"
    assert response.answer == "Grounded answer."
    assert response.generated is True
    assert response.backend == "chroma"
    assert response.timings.first_token_seconds == 0.14
    assert response.sources[0].chunk_id == "overview-3178688:001"


def test_build_graph_neighborhood_response_creates_document_and_chunk_nodes() -> None:
    response = build_graph_neighborhood_response(
        "What changed?",
        [
            SearchResult(
                chunk_id="architecture-3309569:001",
                score=0.91,
                text="The execution layer runs retrieval and generation steps.",
                metadata={
                    "page": "Architecture",
                    "section": "Execution layer",
                    "source_path": "ASA/architecture-3309569.md",
                    "space_key": "ASA",
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
                    "space_key": "ASA",
                },
            ),
        ],
    )

    assert response.query == "What changed?"
    assert response.result_count == 2
    assert response.seed_node_ids == [
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
    ]
    assert {node.id for node in response.nodes} == {
        "document::ASA/architecture-3309569.md",
        "chunk::architecture-3309569:001",
        "chunk::architecture-3309569:002",
    }
    assert [edge.type for edge in response.edges] == ["belongs_to", "belongs_to", "similar_to"]
    assert response.edges[2].metadata["reason"] == "adjacent_retrieval_rank"
