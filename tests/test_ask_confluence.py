"""Tests for the end-to-end ask script helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import ask_confluence


def test_format_duration_uses_two_decimal_places() -> None:
    assert ask_confluence._format_duration(2.3456) == "2.35s"


def test_print_timings_formats_all_stages(capsys) -> None:
    ask_confluence._print_timings(
        embedding_seconds=1.23,
        retrieval_seconds=0.45,
        first_token_seconds=0.89,
        generation_seconds=6.78,
        total_seconds=8.46,
    )

    assert capsys.readouterr().out.splitlines() == [
        "Timings:",
        "embedding: 1.23s",
        "retrieval: 0.45s",
        "first_token: 0.89s",
        "generation: 6.78s",
        "total: 8.46s",
    ]


def test_print_timings_omits_first_token_when_not_available(capsys) -> None:
    ask_confluence._print_timings(
        embedding_seconds=1.0,
        retrieval_seconds=0.5,
        generation_seconds=2.0,
        total_seconds=3.5,
    )

    assert capsys.readouterr().out.splitlines() == [
        "Timings:",
        "embedding: 1.00s",
        "retrieval: 0.50s",
        "generation: 2.00s",
        "total: 3.50s",
    ]


def test_generate_answer_streaming_passes_stream_callback(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_chat_with_ollama(messages: list[dict[str, str]], **kwargs: object):
        captured["messages"] = messages
        captured.update(kwargs)
        token_callback = kwargs["token_callback"]
        assert callable(token_callback)
        token_callback("Grounded answer.")
        return ask_confluence.GenerationResult(
            model="llama3.2:3b",
            content="Grounded answer.",
            first_token_seconds=0.12,
            done_reason="stop",
        )

    monkeypatch.setattr(ask_confluence, "chat_with_ollama", fake_chat_with_ollama)

    result = ask_confluence._generate_answer(
        argparse.Namespace(
            stream=True,
            ollama_host="http://127.0.0.1:11434",
            ollama_model="llama3.2:3b",
            temperature=0.1,
            num_ctx=4096,
            max_tokens=64,
        ),
        [{"role": "user", "content": "Question:\nWhat changed?"}],
    )

    assert captured == {
        "messages": [{"role": "user", "content": "Question:\nWhat changed?"}],
        "host": "http://127.0.0.1:11434",
        "model": "llama3.2:3b",
        "temperature": 0.1,
        "num_ctx": 4096,
        "num_predict": 64,
        "stream": True,
        "token_callback": ask_confluence._stream_token,
    }
    assert result.content == "Grounded answer."
    assert capsys.readouterr().out == "Answer:\nGrounded answer."


def test_main_skips_generation_when_no_results(monkeypatch, capsys, tmp_path: Path) -> None:
    perf_counter_values = iter([100.0, 100.5, 101.0, 101.0, 101.25, 101.5])

    monkeypatch.setattr(ask_confluence, "_configure_console_output", lambda: None)
    monkeypatch.setattr(ask_confluence, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(
        ask_confluence,
        "embed_confluence_query",
        lambda *args, **kwargs: ([0.2, 0.8], SimpleNamespace(collection_name="confluence", backend="chroma")),
    )
    monkeypatch.setattr(
        ask_confluence,
        "retrieve_confluence_context_by_embedding",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        ask_confluence,
        "build_confluence_rag_messages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt building should be skipped")),
    )
    monkeypatch.setattr(
        ask_confluence,
        "_generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generation should be skipped")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ask_confluence.py",
            "What changed?",
            "--persist-dir",
            str(tmp_path / "vector-db"),
            "--prompt",
            str(tmp_path / "prompt.md"),
        ],
    )

    ask_confluence.main()

    assert capsys.readouterr().out.splitlines() == [
        "No relevant context was found. Ollama was not called.",
        "",
        "Timings:",
        "embedding: 0.50s",
        "retrieval: 0.25s",
        "generation: 0.00s",
        "total: 1.50s",
    ]
