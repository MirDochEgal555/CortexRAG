"""Tests for Confluence chunk embedding generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.retrieval.confluence_embeddings import generate_confluence_embeddings


class FakeEncoder:
    """Deterministic stand-in for a sentence embedding model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.model_name_or_path = "fake-mini-model"

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "kwargs": dict(kwargs),
            }
        )
        return [
            [float(index), float(len(text.split()))]
            for index, text in enumerate(texts, start=1)
        ]


def test_generate_confluence_embeddings_writes_embedding_jsonl(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks" / "confluence" / "ASA"
    chunks_dir.mkdir(parents=True)
    output_dir = tmp_path / "storage" / "embeddings"

    chunk_path = chunks_dir / "architecture-3309569.jsonl"
    chunk_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "architecture-3309569:001",
                        "page": "Architecture",
                        "section": "Embeddings",
                        "text": "Embeddings convert text into vectors.",
                        "word_count": 5,
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "architecture-3309569:002",
                        "page": "Architecture",
                        "section": "Retrieval",
                        "text": "Retrieval looks up the nearest chunks.",
                        "word_count": 6,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    encoder = FakeEncoder()
    output_paths = generate_confluence_embeddings(
        input_dir=chunks_dir.parent,
        output_dir=output_dir,
        batch_size=8,
        model_name="ignored-by-fake",
        encoder=encoder,
    )

    embedded_path = output_dir / "ASA" / "architecture-3309569.jsonl"
    assert output_paths == [embedded_path]
    assert embedded_path.exists()

    records = [json.loads(line) for line in embedded_path.read_text(encoding="utf-8").splitlines()]
    assert [record["chunk_id"] for record in records] == [
        "architecture-3309569:001",
        "architecture-3309569:002",
    ]
    assert all(record["embedding_model"] == "fake-mini-model" for record in records)
    assert all(record["embedding_dimensions"] == 2 for record in records)
    assert records[0]["embedding"] == [1.0, 5.0]
    assert records[1]["embedding"] == [2.0, 6.0]

    assert encoder.calls == [
        {
            "texts": [
                "Embeddings convert text into vectors.",
                "Retrieval looks up the nearest chunks.",
            ],
            "kwargs": {
                "batch_size": 8,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            },
        }
    ]


def test_generate_confluence_embeddings_returns_empty_list_for_missing_input_dir(tmp_path: Path) -> None:
    output_paths = generate_confluence_embeddings(
        input_dir=tmp_path / "chunks" / "confluence",
        output_dir=tmp_path / "storage" / "embeddings",
        encoder=FakeEncoder(),
    )

    assert output_paths == []
