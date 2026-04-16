"""Pydantic schemas for the thin UI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cortex_rag.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_NUM_PREDICT,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_RAG_ANSWER_MODE,
    DEFAULT_RAG_PROMPT_PATH,
    DEFAULT_VECTOR_COLLECTION,
    VECTOR_DB_DIR,
)


BackendName = Literal["auto", "chroma", "faiss"]
ResolvedBackendName = Literal["chroma", "faiss"]
AnswerModeName = Literal["concise", "normal", "detailed", "bullet_summary", "technical"]
GraphNodeType = Literal["document", "chunk"]
GraphEdgeType = Literal["belongs_to", "similar_to"]


class APIModel(BaseModel):
    """Shared model configuration for API schemas."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"
    service: str = "cortex_rag"


class SearchRequest(APIModel):
    query: str = Field(min_length=1)
    candidate_k: int = Field(default=10, gt=0)
    top_k: int = Field(default=5, gt=0)
    min_score: float | None = None
    backend: BackendName = "auto"
    collection: str = DEFAULT_VECTOR_COLLECTION
    persist_dir: Path = VECTOR_DB_DIR
    device: str | None = None
    model: str | None = None


class SearchResultPayload(APIModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(APIModel):
    query: str
    result_count: int
    results: list[SearchResultPayload]


class AnswerTimingsPayload(APIModel):
    embedding_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    first_token_seconds: float | None = None


class AnswerRequest(APIModel):
    query: str = Field(min_length=1)
    candidate_k: int = Field(default=10, gt=0)
    top_k: int = Field(default=2, gt=0)
    min_score: float | None = None
    backend: BackendName = "auto"
    collection: str = DEFAULT_VECTOR_COLLECTION
    persist_dir: Path = VECTOR_DB_DIR
    embedding_model: str | None = None
    device: str | None = None
    prompt_path: Path = DEFAULT_RAG_PROMPT_PATH
    answer_mode: AnswerModeName = DEFAULT_RAG_ANSWER_MODE
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE
    num_ctx: int = min(DEFAULT_OLLAMA_NUM_CTX, 4096)
    max_tokens: int = DEFAULT_OLLAMA_NUM_PREDICT


class AnswerResponse(APIModel):
    question: str
    answer: str
    answer_mode: AnswerModeName
    generated: bool
    model: str | None = None
    backend: ResolvedBackendName
    collection_name: str
    prompt_path: str
    sources: list[SearchResultPayload]
    timings: AnswerTimingsPayload


class GraphNodePayload(APIModel):
    id: str
    type: GraphNodeType
    label: str
    highlighted: bool = False
    metadata: dict[str, Any]


class GraphEdgePayload(APIModel):
    id: str
    source: str
    target: str
    type: GraphEdgeType
    weight: float | None = None
    metadata: dict[str, Any]


class GraphNeighborhoodRequest(APIModel):
    query: str = Field(min_length=1)
    candidate_k: int = Field(default=10, gt=0)
    top_k: int = Field(default=5, gt=0)
    min_score: float | None = None
    backend: BackendName = "auto"
    collection: str = DEFAULT_VECTOR_COLLECTION
    persist_dir: Path = VECTOR_DB_DIR
    device: str | None = None
    model: str | None = None


class GraphNeighborhoodResponse(APIModel):
    query: str
    result_count: int
    seed_node_ids: list[str]
    nodes: list[GraphNodePayload]
    edges: list[GraphEdgePayload]
