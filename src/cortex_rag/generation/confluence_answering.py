"""End-to-end Confluence-grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from cortex_rag.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_NUM_PREDICT,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_RAG_ANSWER_MODE,
    DEFAULT_RAG_PROMPT_PATH,
    DEFAULT_VECTOR_COLLECTION,
    VECTOR_DB_DIR,
)
from cortex_rag.generation.ollama_client import GenerationResult, chat_with_ollama
from cortex_rag.generation.prompting import AnswerMode, build_confluence_rag_messages, normalize_answer_mode
from cortex_rag.retrieval import (
    SearchFilters,
    SearchResult,
    embed_confluence_query,
    retrieve_confluence_context_by_embedding,
)
from cortex_rag.retrieval.vector_store import ResolvedBackend, VectorBackend


@dataclass(frozen=True)
class AnswerTimings:
    """Timing breakdown for a single grounded-answer request."""

    embedding_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    first_token_seconds: float | None = None


@dataclass(frozen=True)
class ConfluenceAnswerResult:
    """Structured payload for a grounded-answer request."""

    question: str
    answer_mode: AnswerMode
    prompt_path: Path
    backend: ResolvedBackend
    collection_name: str
    sources: list[SearchResult]
    messages: list[dict[str, str]]
    generation: GenerationResult | None
    timings: AnswerTimings

    @property
    def answer(self) -> str:
        return self.generation.content if self.generation is not None else ""

    @property
    def model(self) -> str | None:
        return self.generation.model if self.generation is not None else None

    @property
    def generated(self) -> bool:
        return self.generation is not None


def answer_confluence_question(
    question: str,
    *,
    candidate_k: int = 10,
    top_k: int = 2,
    min_score: float | None = None,
    backend: VectorBackend = "auto",
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    persist_dir: Path = VECTOR_DB_DIR,
    embedding_model: str | None = None,
    device: str | None = None,
    prompt_path: Path = DEFAULT_RAG_PROMPT_PATH,
    answer_mode: AnswerMode | str = DEFAULT_RAG_ANSWER_MODE,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    max_tokens: int = DEFAULT_OLLAMA_NUM_PREDICT,
    keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
    stream: bool = False,
    token_callback: Callable[[str], None] | None = None,
    clock: Callable[[], float] = perf_counter,
    filters: SearchFilters | None = None,
) -> ConfluenceAnswerResult:
    """Embed, retrieve, prompt, and optionally generate a grounded answer."""

    normalized_mode = normalize_answer_mode(answer_mode)
    total_started_at = clock()

    embedding_started_at = clock()
    query_embedding, manifest = embed_confluence_query(
        question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
        model_name=embedding_model,
        device=device,
    )
    embedding_seconds = clock() - embedding_started_at

    retrieval_started_at = clock()
    sources = retrieve_confluence_context_by_embedding(
        question,
        query_embedding,
        candidate_k=candidate_k,
        final_k=top_k,
        min_score=min_score,
        persist_dir=persist_dir,
        collection_name=manifest.collection_name,
        backend=manifest.backend,
        filters=filters,
    )
    retrieval_seconds = clock() - retrieval_started_at

    messages: list[dict[str, str]] = []
    generation: GenerationResult | None = None
    generation_seconds = 0.0

    if sources:
        messages = build_confluence_rag_messages(
            question,
            sources,
            prompt_path=prompt_path,
            answer_mode=normalized_mode,
        )
        generation_started_at = clock()
        generation = chat_with_ollama(
            messages,
            host=ollama_host,
            model=ollama_model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=max_tokens,
            keep_alive=keep_alive,
            stream=stream,
            token_callback=token_callback,
        )
        generation_seconds = clock() - generation_started_at

    total_seconds = clock() - total_started_at
    return ConfluenceAnswerResult(
        question=question,
        answer_mode=normalized_mode,
        prompt_path=prompt_path,
        backend=manifest.backend,
        collection_name=manifest.collection_name,
        sources=sources,
        messages=messages,
        generation=generation,
        timings=AnswerTimings(
            embedding_seconds=embedding_seconds,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
            first_token_seconds=generation.first_token_seconds if generation is not None else None,
        ),
    )
