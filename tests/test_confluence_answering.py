"""Tests for the package-level grounded-answer flow."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.generation import GenerationResult
from cortex_rag.generation import confluence_answering as answering
from cortex_rag.retrieval import SearchResult


def test_answer_confluence_question_returns_structured_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}
    clock_values = iter([100.0, 100.5, 101.0, 101.0, 101.2, 101.3, 102.3, 102.8])

    def fake_embed_query(question: str, **kwargs: object):
        assert question == "What changed?"
        assert kwargs["collection_name"] == "knowledge"
        assert kwargs["persist_dir"] == Path("storage/chroma")
        return [0.2, 0.8], SimpleNamespace(collection_name="knowledge", backend="chroma")

    def fake_retrieve_by_embedding(question: str, embedding: list[float], **kwargs: object) -> list[SearchResult]:
        assert question == "What changed?"
        assert embedding == [0.2, 0.8]
        assert kwargs["candidate_k"] == 8
        assert kwargs["final_k"] == 3
        assert kwargs["min_score"] == 0.7
        return [
            SearchResult(
                chunk_id="overview-3178688:001",
                score=0.9321,
                text="The agent qualifies and prioritizes leads.",
                metadata={"page": "Overview", "section": "Lead qualification"},
            )
        ]

    def fake_build_messages(question: str, results: list[SearchResult], **kwargs: object) -> list[dict[str, str]]:
        assert question == "What changed?"
        assert [result.chunk_id for result in results] == ["overview-3178688:001"]
        assert kwargs["prompt_path"] == Path("prompts/custom.md")
        assert kwargs["answer_mode"] == "technical"
        return [{"role": "user", "content": "Question:\nWhat changed?"}]

    def fake_chat_with_ollama(messages: list[dict[str, str]], **kwargs: object) -> GenerationResult:
        captured["messages"] = messages
        captured.update(kwargs)
        return GenerationResult(
            model="llama3.2:3b",
            content="Grounded answer.",
            first_token_seconds=0.18,
            prompt_eval_count=42,
            eval_count=11,
            done_reason="stop",
        )

    monkeypatch.setattr(answering, "embed_confluence_query", fake_embed_query)
    monkeypatch.setattr(answering, "retrieve_confluence_context_by_embedding", fake_retrieve_by_embedding)
    monkeypatch.setattr(answering, "build_confluence_rag_messages", fake_build_messages)
    monkeypatch.setattr(answering, "chat_with_ollama", fake_chat_with_ollama)

    result = answering.answer_confluence_question(
        "What changed?",
        candidate_k=8,
        top_k=3,
        min_score=0.7,
        persist_dir=Path("storage/chroma"),
        prompt_path=Path("prompts/custom.md"),
        answer_mode="technical",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3.2:3b",
        temperature=0.1,
        num_ctx=4096,
        max_tokens=64,
        stream=True,
        token_callback=lambda token: None,
        clock=lambda: next(clock_values),
    )

    assert result.question == "What changed?"
    assert result.answer_mode == "technical"
    assert result.backend == "chroma"
    assert result.collection_name == "knowledge"
    assert result.answer == "Grounded answer."
    assert result.model == "llama3.2:3b"
    assert result.generated is True
    assert [source.chunk_id for source in result.sources] == ["overview-3178688:001"]
    assert result.messages == [{"role": "user", "content": "Question:\nWhat changed?"}]
    assert result.timings.embedding_seconds == pytest.approx(0.5)
    assert result.timings.retrieval_seconds == pytest.approx(0.2)
    assert result.timings.generation_seconds == pytest.approx(1.0)
    assert result.timings.total_seconds == pytest.approx(2.8)
    assert result.timings.first_token_seconds == pytest.approx(0.18)
    assert captured == {
        "messages": [{"role": "user", "content": "Question:\nWhat changed?"}],
        "host": "http://127.0.0.1:11434",
        "model": "llama3.2:3b",
        "temperature": 0.1,
        "num_ctx": 4096,
        "num_predict": 64,
        "keep_alive": "5m",
        "stream": True,
        "token_callback": captured["token_callback"],
    }
    assert callable(captured["token_callback"])


def test_answer_confluence_question_skips_generation_without_context(monkeypatch) -> None:
    clock_values = iter([200.0, 200.25, 200.75, 200.75, 201.0, 201.5])

    monkeypatch.setattr(
        answering,
        "embed_confluence_query",
        lambda *args, **kwargs: ([0.2, 0.8], SimpleNamespace(collection_name="knowledge", backend="faiss")),
    )
    monkeypatch.setattr(answering, "retrieve_confluence_context_by_embedding", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        answering,
        "build_confluence_rag_messages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt building should be skipped")),
    )
    monkeypatch.setattr(
        answering,
        "chat_with_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generation should be skipped")),
    )

    result = answering.answer_confluence_question(
        "What changed?",
        clock=lambda: next(clock_values),
    )

    assert result.backend == "faiss"
    assert result.sources == []
    assert result.messages == []
    assert result.generation is None
    assert result.answer == ""
    assert result.model is None
    assert result.generated is False
    assert result.timings.embedding_seconds == pytest.approx(0.5)
    assert result.timings.retrieval_seconds == pytest.approx(0.25)
    assert result.timings.generation_seconds == pytest.approx(0.0)
    assert result.timings.total_seconds == pytest.approx(1.5)
    assert result.timings.first_token_seconds is None
