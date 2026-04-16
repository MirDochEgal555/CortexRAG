"""Pure serialization helpers for the thin UI backend."""

from __future__ import annotations

from pathlib import Path

from cortex_rag.api.schemas import (
    AnswerResponse,
    AnswerTimingsPayload,
    GraphEdgePayload,
    GraphNeighborhoodResponse,
    GraphNodePayload,
    SearchResponse,
    SearchResultPayload,
)
from cortex_rag.generation.confluence_answering import ConfluenceAnswerResult
from cortex_rag.retrieval import SearchResult


def build_search_response(query: str, results: list[SearchResult]) -> SearchResponse:
    """Serialize retrieval hits into the `/search` response shape."""

    return SearchResponse(
        query=query,
        result_count=len(results),
        results=[_search_result_payload(result) for result in results],
    )


def build_answer_response(result: ConfluenceAnswerResult) -> AnswerResponse:
    """Serialize a grounded-answer result into the `/answer` response shape."""

    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        answer_mode=result.answer_mode,
        generated=result.generated,
        model=result.model,
        backend=result.backend,
        collection_name=result.collection_name,
        prompt_path=str(result.prompt_path),
        sources=[_search_result_payload(source) for source in result.sources],
        timings=AnswerTimingsPayload(
            embedding_seconds=result.timings.embedding_seconds,
            retrieval_seconds=result.timings.retrieval_seconds,
            generation_seconds=result.timings.generation_seconds,
            total_seconds=result.timings.total_seconds,
            first_token_seconds=result.timings.first_token_seconds,
        ),
    )


def build_graph_neighborhood_response(
    query: str,
    results: list[SearchResult],
) -> GraphNeighborhoodResponse:
    """Build a graph-ready neighborhood from current retrieval hits.

    This is intentionally thin: it derives document/chunk nodes directly from
    current retrieval results and keeps the edge logic lightweight.
    """

    nodes_by_id: dict[str, GraphNodePayload] = {}
    edges: list[GraphEdgePayload] = []
    seed_node_ids: list[str] = []
    previous_chunk_node_id: str | None = None
    previous_result: SearchResult | None = None

    for result in results:
        chunk_node_id = f"chunk::{result.chunk_id}"
        document_node_id = _document_node_id(result)
        document_label = _document_label(result)

        if document_node_id not in nodes_by_id:
            nodes_by_id[document_node_id] = GraphNodePayload(
                id=document_node_id,
                type="document",
                label=document_label,
                highlighted=True,
                metadata=_document_metadata(result),
            )

        nodes_by_id[chunk_node_id] = GraphNodePayload(
            id=chunk_node_id,
            type="chunk",
            label=_chunk_label(result),
            highlighted=True,
            metadata=_chunk_metadata(result),
        )
        seed_node_ids.append(chunk_node_id)

        edges.append(
            GraphEdgePayload(
                id=f"{document_node_id}--{chunk_node_id}::belongs_to",
                source=document_node_id,
                target=chunk_node_id,
                type="belongs_to",
                weight=1.0,
                metadata={"reason": "chunk_source_membership"},
            )
        )

        if previous_chunk_node_id is not None and previous_result is not None:
            edges.append(
                GraphEdgePayload(
                    id=f"{previous_chunk_node_id}--{chunk_node_id}::similar_to",
                    source=previous_chunk_node_id,
                    target=chunk_node_id,
                    type="similar_to",
                    weight=(previous_result.score + result.score) / 2.0,
                    metadata={"reason": "adjacent_retrieval_rank"},
                )
            )

        previous_chunk_node_id = chunk_node_id
        previous_result = result

    return GraphNeighborhoodResponse(
        query=query,
        result_count=len(results),
        seed_node_ids=seed_node_ids,
        nodes=list(nodes_by_id.values()),
        edges=edges,
    )


def _search_result_payload(result: SearchResult) -> SearchResultPayload:
    return SearchResultPayload(
        chunk_id=result.chunk_id,
        score=result.score,
        text=result.text,
        metadata=dict(result.metadata),
    )


def _document_node_id(result: SearchResult) -> str:
    source_path = _metadata_text(result, "source_path")
    if source_path:
        return f"document::{source_path}"

    page = _metadata_text(result, "page")
    if page:
        return f"document::{page}"

    return f"document::{result.chunk_id.split(':', 1)[0]}"


def _document_label(result: SearchResult) -> str:
    page = _metadata_text(result, "page")
    if page:
        return page

    source_path = _metadata_text(result, "source_path")
    if source_path:
        return Path(source_path).name

    return result.chunk_id.split(":", 1)[0]


def _chunk_label(result: SearchResult) -> str:
    section = _metadata_text(result, "section")
    if section:
        return section
    return result.chunk_id


def _document_metadata(result: SearchResult) -> dict[str, object]:
    return {
        "page": _metadata_text(result, "page"),
        "source_path": _metadata_text(result, "source_path"),
        "source": _metadata_text(result, "source"),
        "space_key": _metadata_text(result, "space_key"),
    }


def _chunk_metadata(result: SearchResult) -> dict[str, object]:
    metadata = dict(result.metadata)
    metadata["retrieval_score"] = result.score
    return metadata


def _metadata_text(result: SearchResult, key: str) -> str:
    value = result.metadata.get(key)
    return str(value).strip() if value not in (None, "") else ""
