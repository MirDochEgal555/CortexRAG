"""Generate embeddings for Confluence chunk JSONL files."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any, Protocol

from cortex_rag.config import CHUNKS_DIR, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR


CONFLUENCE_CHUNKS_DIR = CHUNKS_DIR / "confluence"
CONFLUENCE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "confluence"


class TextEncoder(Protocol):
    """Minimal protocol for sentence embedding backends."""

    def encode(self, texts: Sequence[str], **kwargs: object) -> Any:
        """Return one vector per input text."""


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

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    output_paths: list[Path] = []
    if not input_dir.exists():
        return output_paths

    embedding_model = model_name
    active_encoder = encoder
    if active_encoder is None:
        active_encoder = _load_sentence_transformer(model_name=model_name, device=device)
    embedding_model = str(getattr(active_encoder, "model_name_or_path", model_name))

    for space_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        output_paths.extend(
            generate_confluence_space_embeddings(
                space_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                model_name=embedding_model,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                encoder=active_encoder,
            )
        )

    return output_paths


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

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    chunk_paths = sorted(space_dir.glob("*.jsonl"))
    if not chunk_paths:
        return []

    active_encoder = encoder or _load_sentence_transformer(model_name=model_name, device=None)
    embedding_model = str(getattr(active_encoder, "model_name_or_path", model_name))

    space_output_dir = output_dir / space_dir.name
    space_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for chunk_path in chunk_paths:
        chunks = _load_chunk_records(chunk_path)
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        vectors = _encode_texts(
            active_encoder,
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

        records = [
            {
                **chunk,
                "embedding_model": embedding_model,
                "embedding_dimensions": len(vector),
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        output_path = space_output_dir / chunk_path.name
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def _load_sentence_transformer(*, model_name: str, device: str | None) -> TextEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings. "
            "Install project dependencies before running this step."
        ) from exc

    kwargs: dict[str, object] = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


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


def _encode_texts(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
) -> list[list[float]]:
    if not texts:
        return []

    try:
        encoded = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
    except TypeError:
        encoded = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    vectors = [_vector_to_list(vector) for vector in encoded]
    if len(vectors) != len(texts):
        raise ValueError("Encoder returned a different number of vectors than input texts.")
    return vectors


def _vector_to_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = vector

    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise TypeError("Embedding vector is not iterable.")

    return [float(value) for value in values]
