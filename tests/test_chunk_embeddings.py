"""Tests for source-neutral chunk embedding generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.retrieval.chunk_embeddings import generate_knowledge_embeddings


class FakeEncoder:
    """Deterministic stand-in for a sentence embedding model."""

    model_name_or_path = "fake-mini-model"

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        return [[float(index), float(len(text.split()))] for index, text in enumerate(texts, start=1)]


def test_generate_knowledge_embeddings_writes_obsidian_and_zotero_records(tmp_path: Path) -> None:
    obsidian_dir = tmp_path / "chunks" / "obsidian"
    zotero_dir = tmp_path / "chunks" / "zotero"
    output_dir = tmp_path / "embeddings" / "knowledge"
    obsidian_dir.joinpath("main").mkdir(parents=True)
    zotero_dir.mkdir(parents=True)

    obsidian_dir.joinpath("main", "Research.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "obsidian::main::research:001",
                "document_id": "obsidian::main::research",
                "source": "obsidian",
                "text": "Obsidian note text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    zotero_dir.joinpath("paper.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "zotero::doe2024rag:001",
                "document_id": "zotero::doe2024rag",
                "source": "zotero",
                "text": "Zotero paper text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_paths = generate_knowledge_embeddings(
        input_dirs=[obsidian_dir, zotero_dir],
        output_dir=output_dir,
        encoder=FakeEncoder(),
    )

    assert output_paths == [
        output_dir / "obsidian" / "main" / "Research.jsonl",
        output_dir / "zotero" / "paper.jsonl",
    ]
    records = [
        json.loads(line)
        for path in output_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["source"] for record in records] == ["obsidian", "zotero"]
    assert all(record["embedding_model"] == "fake-mini-model" for record in records)
    assert all(record["embedding_dimensions"] == 2 for record in records)
