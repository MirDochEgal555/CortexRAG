"""Generate embeddings for source-neutral chunk JSONL trees."""

from __future__ import annotations

import json
from pathlib import Path

from cortex_rag.config import CHUNKS_DIR, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR
from cortex_rag.retrieval.embedding_utils import (
    TextEncoder,
    encode_texts,
    load_sentence_transformer,
)


KNOWLEDGE_CHUNKS_DIRS = (CHUNKS_DIR / "obsidian", CHUNKS_DIR / "zotero")
KNOWLEDGE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "knowledge"


def generate_chunk_embeddings(
    input_dirs: list[Path],
    output_dir: Path = KNOWLEDGE_EMBEDDINGS_DIR,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
    include_source_dir: bool = True,
) -> list[Path]:
    """Embed every chunk JSONL file under one or more source chunk directories."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    existing_dirs = [input_dir for input_dir in input_dirs if input_dir.exists()]
    if not existing_dirs:
        return []

    active_encoder = encoder or load_sentence_transformer(model_name=model_name, device=device)
    embedding_model = str(getattr(active_encoder, "model_name_or_path", model_name))

    output_paths: list[Path] = []
    for input_dir in existing_dirs:
        source_name = input_dir.name
        for chunk_path in sorted(input_dir.rglob("*.jsonl")):
            relative_path = chunk_path.relative_to(input_dir)
            output_path = output_dir / source_name / relative_path if include_source_dir else output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records = _load_chunk_records(chunk_path)
            embedded_records = _embed_records(
                records,
                encoder=active_encoder,
                embedding_model=embedding_model,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            lines = [json.dumps(record, ensure_ascii=False) for record in embedded_records]
            output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            output_paths.append(output_path)

    return output_paths


def generate_knowledge_embeddings(
    input_dirs: list[Path] | None = None,
    output_dir: Path = KNOWLEDGE_EMBEDDINGS_DIR,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
) -> list[Path]:
    """Embed the default Zotero and Obsidian chunk trees for the UI knowledge index."""

    return generate_chunk_embeddings(
        list(input_dirs or KNOWLEDGE_CHUNKS_DIRS),
        output_dir=output_dir,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        encoder=encoder,
        include_source_dir=True,
    )


def _load_chunk_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Chunk file contains a non-object record: {path}")
        records.append(payload)
    return records


def _embed_records(
    records: list[dict[str, object]],
    *,
    encoder: TextEncoder,
    embedding_model: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> list[dict[str, object]]:
    texts = [str(record.get("text", "")) for record in records]
    vectors = encode_texts(
        encoder,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    embedded_records: list[dict[str, object]] = []
    for record, vector in zip(records, vectors, strict=True):
        embedded_records.append(
            {
                **record,
                "embedding_model": embedding_model,
                "embedding_dimensions": len(vector),
                "embedding": vector,
            }
        )
    return embedded_records
