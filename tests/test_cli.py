"""Tests for the CortexRAG CLI entry points."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag import cli
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
