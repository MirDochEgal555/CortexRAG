"""Generate embeddings for Confluence chunk JSONL files."""

from __future__ import annotations

from pathlib import Path

from cortex_rag.config import CHUNKS_DIR, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR
from cortex_rag.retrieval.chunk_embeddings import generate_chunk_embeddings
from cortex_rag.retrieval.embedding_utils import TextEncoder


CONFLUENCE_CHUNKS_DIR = CHUNKS_DIR / "confluence"
CONFLUENCE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "confluence"


def generate_confluence_embeddings(
    input_dir: Path = CONFLUENCE_CHUNKS_DIR,
    output_dir: Path = CONFLUENCE_EMBEDDINGS_DIR,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
) -> list[Path]:
    """Embed every chunk JSONL file produced from processed Confluence pages."""

    return generate_chunk_embeddings(
        [input_dir],
        output_dir=output_dir,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        encoder=encoder,
        include_source_dir=False,
    )


def generate_confluence_space_embeddings(
    space_dir: Path,
    *,
    input_dir: Path = CONFLUENCE_CHUNKS_DIR,
    output_dir: Path = CONFLUENCE_EMBEDDINGS_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    encoder: TextEncoder | None = None,
) -> list[Path]:
    """Embed all chunk JSONL files inside one Confluence space directory."""

    return generate_chunk_embeddings(
        [space_dir],
        output_dir=output_dir / space_dir.relative_to(input_dir),
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        encoder=encoder,
        include_source_dir=False,
    )
