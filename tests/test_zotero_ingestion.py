"""Tests for Zotero ingestion."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.zotero import chunk_zotero_items, preprocess_zotero_export


def test_preprocess_zotero_export_writes_normalized_markdown_with_notes(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "zotero"
    raw.mkdir(parents=True)
    bib = raw / "library.bib"
    bib.write_text(
        """
@article{doe2024rag,
  title = {Retrieval-Augmented Generation for Knowledge Work},
  author = {Doe, Jane and Smith, Alex},
  year = {2024},
  doi = {10.1234/example},
  keywords = {rag, evaluation},
  abstract = {A study about retrieval augmented generation.}
}
""".strip(),
        encoding="utf-8",
    )
    notes = raw / "notes"
    notes.mkdir()
    notes.joinpath("doe2024rag.md").write_text("Important annotation text.", encoding="utf-8")

    output_paths = preprocess_zotero_export(bib, tmp_path / "processed" / "zotero")

    output_path = tmp_path / "processed" / "zotero" / "retrieval-augmented-generation-for-knowledge-work.md"
    assert output_paths == [output_path]
    text = output_path.read_text(encoding="utf-8")
    assert 'source: "zotero"' in text
    assert 'citekey: "doe2024rag"' in text
    assert "Jane" in text
    assert "## Abstract" in text
    assert "Important annotation text." in text


def test_chunk_zotero_items_emits_bibliographic_metadata(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "zotero"
    processed.mkdir(parents=True)
    processed.joinpath("doe2024rag.md").write_text(
        "\n".join(
            [
                "---",
                'source: "zotero"',
                'zotero_key: "ABCD1234"',
                'citekey: "doe2024rag"',
                'page_title: "RAG for Knowledge Work"',
                'title: "RAG for Knowledge Work"',
                "authors:",
                '  - "Jane Doe"',
                "year: 2024",
                'doi: "10.1234/example"',
                "tags:",
                '  - "rag"',
                "---",
                "",
                "# RAG for Knowledge Work",
                "",
                "## Notes",
                "",
                " ".join(f"note{index}" for index in range(230)),
            ]
        ),
        encoding="utf-8",
    )

    output_paths = chunk_zotero_items(processed, tmp_path / "chunks" / "zotero")

    chunks_path = tmp_path / "chunks" / "zotero" / "doe2024rag.jsonl"
    assert output_paths == [chunks_path]
    chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert chunk["chunk_id"] == "zotero::doe2024rag:001"
    assert chunk["document_id"] == "zotero::doe2024rag"
    assert chunk["source"] == "zotero"
    assert chunk["page"] == "RAG for Knowledge Work"
    assert chunk["source_path"] == "doe2024rag.md"
    assert chunk["metadata"]["authors"] == ["Jane Doe"]
    assert chunk["metadata"]["year"] == 2024
    assert chunk["metadata"]["doi"] == "10.1234/example"
    assert chunk["metadata"]["citekey"] == "doe2024rag"
