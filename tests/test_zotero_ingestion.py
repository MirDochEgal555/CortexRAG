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


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, path: Path) -> None:
        assert path.name == "doe2024rag.pdf"
        self.pages = [
            _FakePdfPage("First PDF page text about retrieval.\nWith line breaks."),
            _FakePdfPage("Second PDF page text about evaluation."),
        ]


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


def test_preprocess_zotero_export_extracts_matched_pdf_attachment_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cortex_rag.ingestion.zotero as zotero_module

    monkeypatch.setattr(zotero_module, "PdfReader", _FakePdfReader)

    raw = tmp_path / "raw" / "zotero"
    raw.mkdir(parents=True)
    bib = raw / "library.bib"
    bib.write_text(
        """
@article{doe2024rag,
  title = {Retrieval-Augmented Generation for Knowledge Work},
  author = {Doe, Jane},
  year = {2024}
}
""".strip(),
        encoding="utf-8",
    )
    attachments = raw / "attachments"
    attachments.mkdir()
    attachments.joinpath("doe2024rag.pdf").write_bytes(b"%PDF test fixture")

    output_paths = preprocess_zotero_export(bib, tmp_path / "processed" / "zotero")

    text = output_paths[0].read_text(encoding="utf-8")
    assert 'attachment_paths:\n  - "doe2024rag.pdf"' in text
    assert 'extracted_attachment_paths:\n  - "doe2024rag.pdf"' in text
    assert 'attachment_page_counts: {"doe2024rag.pdf": 2}' in text
    assert "_attachment_texts" not in text
    assert "## Attachment Text" in text
    assert "### doe2024rag.pdf" in text
    assert "#### Page 1" in text
    assert "First PDF page text about retrieval. With line breaks." in text
    assert "#### Page 2" in text
    assert "Second PDF page text about evaluation." in text


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
                "attachment_paths:",
                '  - "doe2024rag.pdf"',
                "extracted_attachment_paths:",
                '  - "doe2024rag.pdf"',
                'attachment_page_counts: {"doe2024rag.pdf": 2}',
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
    assert chunk["metadata"]["attachment_paths"] == ["doe2024rag.pdf"]
    assert chunk["metadata"]["extracted_attachment_paths"] == ["doe2024rag.pdf"]
    assert chunk["metadata"]["attachment_page_counts"] == {"doe2024rag.pdf": 2}
