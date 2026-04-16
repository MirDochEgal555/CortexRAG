"""Tests for Ollama generation helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.generation import (
    AnswerMode,
    GenerationResult,
    build_confluence_rag_messages,
    chat_with_ollama,
    format_retrieval_context,
    load_system_prompt,
    normalize_answer_mode,
)
from cortex_rag.retrieval import SearchResult


def test_load_system_prompt_rejects_missing_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError, match="Prompt template does not exist"):
        load_system_prompt(prompt_path)


def test_load_system_prompt_rejects_empty_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="Prompt template is empty"):
        load_system_prompt(prompt_path)


def test_format_retrieval_context_includes_metadata() -> None:
    results = [
        SearchResult(
            chunk_id="architecture-3309569:001",
            score=0.9123,
            text="The execution layer runs retrieval and generation steps.",
            metadata={"page": "Architecture", "section": "Execution layer"},
        )
    ]

    assert format_retrieval_context(results) == "\n".join(
        [
            "Source 1",
            "Chunk ID: architecture-3309569:001",
            "Page: Architecture",
            "Section: Execution layer",
            "Score: 0.9123",
            "Text:",
            "The execution layer runs retrieval and generation steps.",
        ]
    )


def test_format_retrieval_context_returns_placeholder_without_results() -> None:
    assert format_retrieval_context([]) == "No retrieved context was available."


def test_format_retrieval_context_defaults_missing_metadata_and_trims_text() -> None:
    results = [
        SearchResult(
            chunk_id="overview-3178688:001",
            score=0.5,
            text="  Lead routing happens after qualification.  \n",
            metadata={"page": " ", "section": None},
        )
    ]

    assert format_retrieval_context(results) == "\n".join(
        [
            "Source 1",
            "Chunk ID: overview-3178688:001",
            "Page: Unknown page",
            "Section: Unspecified section",
            "Score: 0.5000",
            "Text:",
            "Lead routing happens after qualification.",
        ]
    )


def test_build_confluence_rag_messages_uses_prompt_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Answer only from context.", encoding="utf-8")

    messages = build_confluence_rag_messages(
        "What does the execution layer do?",
        [
            SearchResult(
                chunk_id="architecture-3309569:001",
                score=0.8,
                text="The execution layer orchestrates retrieval and generation.",
                metadata={"page": "Architecture", "section": "Execution layer"},
            )
        ],
        prompt_path=prompt_path,
        answer_mode="technical",
    )

    assert messages[0] == {"role": "system", "content": "Answer only from context."}
    assert "Question:\nWhat does the execution layer do?" in messages[1]["content"]
    assert (
        "Answer mode: technical. Use a technical style, emphasizing implementation details, "
        "structure, and precise terminology."
    ) in messages[1]["content"]
    assert "Chunk ID: architecture-3309569:001" in messages[1]["content"]


def test_build_confluence_rag_messages_rejects_blank_question(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Answer only from context.", encoding="utf-8")

    with pytest.raises(ValueError, match="question must not be empty"):
        build_confluence_rag_messages("   ", [], prompt_path=prompt_path)


def test_normalize_answer_mode_accepts_supported_modes() -> None:
    mode: AnswerMode = normalize_answer_mode("Detailed")
    assert mode == "detailed"


def test_normalize_answer_mode_rejects_unknown_modes() -> None:
    try:
        normalize_answer_mode("poetic")
    except ValueError as exc:
        assert "Unsupported answer mode" in str(exc)
    else:
        raise AssertionError("Expected normalize_answer_mode to reject unsupported modes.")


def test_chat_with_ollama_passes_expected_request_options() -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        model = "llama3.2:3b"
        message = type("Message", (), {"content": "Grounded answer."})()
        prompt_eval_count = 42
        eval_count = 11
        done_reason = "stop"

    class FakeClient:
        def __init__(self, *, host: str) -> None:
            captured["host"] = host

        def chat(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    result = chat_with_ollama(
        [
            {"role": "system", "content": "Answer only from context."},
            {"role": "user", "content": "Question:\nWhat is the architecture?"},
        ],
        host="http://127.0.0.1:11434",
        model="llama3.2:3b",
        temperature=0.1,
        num_ctx=4096,
        num_predict=128,
        keep_alive="10m",
        client_factory=FakeClient,
    )

    assert result == GenerationResult(
        model="llama3.2:3b",
        content="Grounded answer.",
        first_token_seconds=0.0,
        prompt_eval_count=42,
        eval_count=11,
        done_reason="stop",
    )
    assert captured == {
        "host": "http://127.0.0.1:11434",
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "Answer only from context."},
            {"role": "user", "content": "Question:\nWhat is the architecture?"},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 128,
        },
        "keep_alive": "10m",
    }


def test_chat_with_ollama_streams_and_reports_first_token() -> None:
    streamed_tokens: list[str] = []

    class FakeResponse:
        def __init__(
            self,
            *,
            content: str,
            done_reason: str | None = None,
            prompt_eval_count: int | None = None,
            eval_count: int | None = None,
        ) -> None:
            self.model = "llama3.2:3b"
            self.message = type("Message", (), {"content": content})()
            self.done_reason = done_reason
            self.prompt_eval_count = prompt_eval_count
            self.eval_count = eval_count

    class FakeClient:
        def __init__(self, *, host: str) -> None:
            self.host = host

        def chat(self, **kwargs: object):
            assert kwargs["stream"] is True
            return iter(
                [
                    FakeResponse(content="Grounded "),
                    FakeResponse(
                        content="answer.",
                        done_reason="stop",
                        prompt_eval_count=21,
                        eval_count=7,
                    ),
                ]
            )

    result = chat_with_ollama(
        [{"role": "user", "content": "Question:\nWhat is the architecture?"}],
        stream=True,
        token_callback=streamed_tokens.append,
        client_factory=FakeClient,
    )

    assert result.model == "llama3.2:3b"
    assert result.content == "Grounded answer."
    assert result.first_token_seconds is not None
    assert result.first_token_seconds >= 0.0
    assert result.prompt_eval_count == 21
    assert result.eval_count == 7
    assert result.done_reason == "stop"
    assert streamed_tokens == ["Grounded ", "answer."]


def test_chat_with_ollama_streaming_without_tokens_returns_empty_result() -> None:
    class FakeClient:
        def __init__(self, *, host: str) -> None:
            self.host = host

        def chat(self, **kwargs: object):
            assert kwargs["stream"] is True
            return iter([])

    result = chat_with_ollama(
        [{"role": "user", "content": "Question:\nWhat is the architecture?"}],
        model="fallback-model",
        stream=True,
        client_factory=FakeClient,
    )

    assert result == GenerationResult(
        model="fallback-model",
        content="",
        first_token_seconds=None,
        prompt_eval_count=None,
        eval_count=None,
        done_reason=None,
    )
