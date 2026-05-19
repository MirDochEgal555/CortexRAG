"""Tests for Obsidian ingestion."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.obsidian import chunk_obsidian_notes, preprocess_obsidian_vault


def test_preprocess_obsidian_vault_preserves_metadata_and_normalizes_wikilinks(tmp_path: Path) -> None:
    vault = tmp_path / "raw" / "obsidian" / "main"
    vault.mkdir(parents=True)
    note = vault / "Projects" / "CortexRAG.md"
    note.parent.mkdir()
    note.write_text(
        "\n".join(
            [
                "---",
                'title: "CortexRAG Plan"',
                "tags:",
                '  - "rag"',
                "---",
                "",
                "Some text with #project/foo and [[Target Note|target label]].",
                "[External](https://example.com)",
            ]
        ),
        encoding="utf-8",
    )
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "workspace.md").write_text("skip me", encoding="utf-8")

    output_paths = preprocess_obsidian_vault(vault, tmp_path / "processed" / "obsidian")

    output_path = tmp_path / "processed" / "obsidian" / "main" / "Projects" / "CortexRAG.md"
    assert output_paths == [output_path]
    text = output_path.read_text(encoding="utf-8")
    assert 'vault_name: "main"' in text
    assert 'page_title: "CortexRAG Plan"' in text
    assert '"target": "Target Note"' in text
    assert "target label" in text
    assert "[[Target Note|target label]]" not in text
    assert "# CortexRAG Plan" in text


def test_chunk_obsidian_notes_emits_common_contract(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "obsidian" / "main"
    processed.mkdir(parents=True)
    processed.joinpath("Research.md").write_text(
        "\n".join(
            [
                "---",
                'source: "obsidian"',
                'vault_name: "main"',
                'page_title: "Research"',
                'source_path: "Research.md"',
                "tags:",
                '  - "rag"',
                "wikilinks:",
                '  - {"target": "Literature", "label": "Literature"}',
                "---",
                "",
                "# Research",
                "",
                "## Notes",
                "",
                " ".join(f"word{index}" for index in range(230)),
            ]
        ),
        encoding="utf-8",
    )

    output_paths = chunk_obsidian_notes(tmp_path / "processed" / "obsidian", tmp_path / "chunks" / "obsidian")

    chunks_path = tmp_path / "chunks" / "obsidian" / "main" / "Research.jsonl"
    assert output_paths == [chunks_path]
    chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert chunk["chunk_id"] == "obsidian::main::research:001"
    assert chunk["document_id"] == "obsidian::main::research"
    assert chunk["source"] == "obsidian"
    assert chunk["page"] == "Research"
    assert chunk["section"] == "Notes"
    assert chunk["source_path"] == "main/Research.md"
    assert chunk["metadata"]["vault_name"] == "main"
    assert chunk["metadata"]["tags"] == ["rag"]
