"""FastAPI application for the thin UI backend."""

from __future__ import annotations

from threading import Lock
from typing import Any

from cortex_rag.config import DEFAULT_VECTOR_COLLECTION, VECTOR_DB_DIR
from cortex_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from cortex_rag.api.serializers import (
    build_answer_response,
    build_graph_neighborhood_response,
    build_search_response,
)
from cortex_rag.generation import answer_confluence_question
from cortex_rag.graph import build_graph_neighborhood, load_confluence_graph
from cortex_rag.index_status import verify_current_searchable_index
from cortex_rag.retrieval import (
    SearchFilters,
    preload_sentence_transformer,
    retrieve_confluence_context,
)


_UI_RUNTIME_WARMUP_LOCK = Lock()
_WARMED_UI_RUNTIME_KEYS: set[tuple[str, str]] = set()


def warm_ui_runtime_assets(
    *,
    persist_dir=VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
) -> None:
    """Verify the UI index and preload the embedding model once per process."""
    cache_key = (str(persist_dir), collection_name)
    if cache_key in _WARMED_UI_RUNTIME_KEYS:
        return

    with _UI_RUNTIME_WARMUP_LOCK:
        if cache_key in _WARMED_UI_RUNTIME_KEYS:
            return
        index_status = verify_current_searchable_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        preload_sentence_transformer(
            model_name=index_status.embedding_model,
            device=None,
        )
        _WARMED_UI_RUNTIME_KEYS.add(cache_key)


def create_app() -> Any:
    """Create the FastAPI app lazily so the package stays importable without FastAPI."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install FastAPI to run the CortexRAG UI backend."
        ) from exc

    app = FastAPI(title="CortexRAG API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            warm_ui_runtime_assets()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return HealthResponse()

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            warm_ui_runtime_assets(
                persist_dir=request.persist_dir,
                collection_name=request.collection,
            )
            results = retrieve_confluence_context(
                request.query,
                candidate_k=request.candidate_k,
                final_k=request.top_k,
                min_score=request.min_score,
                persist_dir=request.persist_dir,
                collection_name=request.collection,
                backend=request.backend,
                model_name=request.model,
                device=request.device,
                filters=_filters_from_request(request),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_search_response(request.query, results)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        try:
            warm_ui_runtime_assets(
                persist_dir=request.persist_dir,
                collection_name=request.collection,
            )
            result = answer_confluence_question(
                request.query,
                candidate_k=request.candidate_k,
                top_k=request.top_k,
                min_score=request.min_score,
                backend=request.backend,
                collection_name=request.collection,
                persist_dir=request.persist_dir,
                embedding_model=request.embedding_model,
                device=request.device,
                prompt_path=request.prompt_path,
                answer_mode=request.answer_mode,
                ollama_host=request.ollama_host,
                ollama_model=request.ollama_model,
                temperature=request.temperature,
                num_ctx=request.num_ctx,
                max_tokens=request.max_tokens,
                filters=_filters_from_request(request),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_answer_response(result)

    @app.post("/graph/neighborhood", response_model=GraphNeighborhoodResponse)
    def graph_neighborhood(request: GraphNeighborhoodRequest) -> GraphNeighborhoodResponse:
        try:
            warm_ui_runtime_assets(
                persist_dir=request.persist_dir,
                collection_name=request.collection,
            )
            results = retrieve_confluence_context(
                request.query,
                candidate_k=request.candidate_k,
                final_k=request.top_k,
                min_score=request.min_score,
                persist_dir=request.persist_dir,
                collection_name=request.collection,
                backend=request.backend,
                model_name=request.model,
                device=request.device,
                filters=_filters_from_request(request),
            )
            graph = load_confluence_graph(
                persist_dir=request.persist_dir,
                collection_name=request.collection,
            )
            request_filters = _filters_from_request(request)
            neighborhood = build_graph_neighborhood(
                graph,
                seed_chunk_ids=[result.chunk_id for result in results],
                allowed_chunk_ids=[result.chunk_id for result in results] if request_filters else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_graph_neighborhood_response(request.query, neighborhood)

    return app


def _filters_from_request(request: SearchRequest | AnswerRequest | GraphNeighborhoodRequest) -> SearchFilters | None:
    filters = SearchFilters(
        source=request.source,
        document_id=request.document_id,
        citekey=request.citekey,
        doi=request.doi,
        title=request.title,
        zotero_key=request.zotero_key,
    )
    return filters if filters.active else None
