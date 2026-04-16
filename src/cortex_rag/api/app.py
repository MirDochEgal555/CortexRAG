"""FastAPI application for the thin UI backend."""

from __future__ import annotations

from typing import Any

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
from cortex_rag.retrieval import retrieve_confluence_context


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
        return HealthResponse()

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
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
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_search_response(request.query, results)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        try:
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
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_answer_response(result)

    @app.post("/graph/neighborhood", response_model=GraphNeighborhoodResponse)
    def graph_neighborhood(request: GraphNeighborhoodRequest) -> GraphNeighborhoodResponse:
        try:
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
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return build_graph_neighborhood_response(request.query, results)

    return app
