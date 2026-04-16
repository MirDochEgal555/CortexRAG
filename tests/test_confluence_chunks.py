"""Tests for chunking processed Confluence Markdown pages."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.confluence_chunks import chunk_confluence_exports


def test_chunk_confluence_exports_writes_heading_aware_jsonl(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed" / "confluence" / "ASA"
    processed_dir.mkdir(parents=True)
    chunks_dir = tmp_path / "chunks"

    architecture_page = processed_dir / "rag-architecture-111.md"
    retrieval_page = processed_dir / "retrieval-222.md"

    architecture_page.write_text(
        _build_page(
            page_title="RAG Architecture",
            source_html="ASA/RAG-Architecture_111.html",
            body="\n".join(
                [
                    "# RAG Architecture",
                    "",
                    _paragraph("overview", 40),
                    "",
                    "## Embeddings",
                    "",
                    _paragraph("embeddings", 70),
                    "",
                    f"[Retrieval]({retrieval_page.name}) " + _paragraph("bridge", 8),
                    "",
                    _paragraph("vectors", 140),
                    "",
                    "## Retrieval Flow",
                    "",
                    _paragraph("retrieval", 220),
                ]
            ),
        ),
        encoding="utf-8",
    )
    retrieval_page.write_text(
        _build_page(
            page_title="Retrieval",
            source_html="ASA/Retrieval_222.html",
            body="\n".join(
                [
                    "# Retrieval",
                    "",
                    "## Overview",
                    "",
                    _paragraph("lookup", 230),
                ]
            ),
        ),
        encoding="utf-8",
    )

    output_paths = chunk_confluence_exports(processed_dir.parent, chunks_dir)

    architecture_chunks_path = chunks_dir / "ASA" / "rag-architecture-111.jsonl"
    assert architecture_chunks_path in output_paths
    assert architecture_chunks_path.exists()

    chunks = [json.loads(line) for line in architecture_chunks_path.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == 2

    embeddings_chunk = chunks[0]
    assert embeddings_chunk["page"] == "RAG Architecture"
    assert embeddings_chunk["section"] == "Embeddings"
    assert embeddings_chunk["headings"] == ["RAG Architecture", "Embeddings"]
    assert embeddings_chunk["source"] == "confluence"
    assert embeddings_chunk["source_path"] == "ASA/rag-architecture-111.md"
    assert 200 <= embeddings_chunk["word_count"] <= 500
    assert embeddings_chunk["links"] == [
        {
            "text": "Retrieval",
            "target_path": "retrieval-222.md",
            "target_page": "Retrieval",
        }
    ]
    assert "Retrieval" in embeddings_chunk["text"]

    retrieval_flow_chunk = chunks[1]
    assert retrieval_flow_chunk["section"] == "Retrieval Flow"
    assert retrieval_flow_chunk["headings"] == ["RAG Architecture", "Retrieval Flow"]
    assert 200 <= retrieval_flow_chunk["word_count"] <= 500


def test_chunk_confluence_exports_merges_small_sibling_sections_under_parent_heading(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed" / "confluence" / "ASA"
    processed_dir.mkdir(parents=True)
    chunks_dir = tmp_path / "chunks"

    page_path = processed_dir / "lead-scoring-333.md"
    page_path.write_text(
        _build_page(
            page_title="Lead Scoring",
            source_html="ASA/Lead-Scoring_333.html",
            body="\n".join(
                [
                    "# Lead Scoring",
                    "",
                    "## Signals",
                    "",
                    _paragraph("signals", 75),
                    "",
                    "## Thresholds",
                    "",
                    _paragraph("thresholds", 75),
                    "",
                    "## Routing",
                    "",
                    _paragraph("routing", 75),
                ]
            ),
        ),
        encoding="utf-8",
    )

    chunk_confluence_exports(processed_dir.parent, chunks_dir)

    chunks_path = chunks_dir / "ASA" / "lead-scoring-333.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]

    assert len(chunks) == 1
    assert chunks[0]["page"] == "Lead Scoring"
    assert chunks[0]["section"] == "Lead Scoring"
    assert chunks[0]["headings"] == ["Lead Scoring"]
    assert 200 <= chunks[0]["word_count"] <= 500
    assert "Signals" in chunks[0]["text"]
    assert "Thresholds" in chunks[0]["text"]
    assert "Routing" in chunks[0]["text"]


def _build_page(*, page_title: str, source_html: str, body: str) -> str:
    return "\n".join(
        [
            "---",
            'space_key: "ASA"',
            'space_name: "AI Sales Agent"',
            f'page_title: "{page_title}"',
            'page_type: "page"',
            'source_zip: "ASA_2026-04-16.zip"',
            f'source_html: "{source_html}"',
            "breadcrumbs:",
            '  - "AI Sales Agent"',
            'created_by: "Robin Keim"',
            'created_on: "2026-03-28"',
            "---",
            "",
            body,
            "",
        ]
    )


def _paragraph(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))
