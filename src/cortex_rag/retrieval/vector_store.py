"""Persistent vector-store build and search helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, cast

from cortex_rag.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_COLLECTION,
    EMBEDDINGS_DIR,
    VECTOR_DB_DIR,
)
from cortex_rag.retrieval.embedding_utils import TextEncoder, encode_texts, load_sentence_transformer


VectorBackend = Literal["auto", "chroma", "faiss"]
ResolvedBackend = Literal["chroma", "faiss"]
CONFLUENCE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "confluence"
_NORMALIZE_TEXT_PATTERN = re.compile(r"[^a-z0-9]+")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_PAGE_AGREEMENT_BOOST = 0.05
_MAX_PAGE_AGREEMENT_BOOST = 0.15
_SECTION_KEYWORD_BOOST = 0.03
_MAX_SECTION_KEYWORD_BOOST = 0.12
_NEAR_DUPLICATE_OVERLAP_THRESHOLD = 0.85
_MIN_OVERLAP_TOKENS = 5


@dataclass(frozen=True)
class VectorStoreBuildResult:
    """Details about a completed vector-store build."""

    backend: ResolvedBackend
    collection_name: str
    persist_dir: Path
    document_count: int
    embedding_dimensions: int
    embedding_model: str
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class VectorStoreManifest:
    """Persisted metadata describing the built vector store."""

    backend: ResolvedBackend
    collection_name: str
    document_count: int
    embedding_dimensions: int
    embedding_model: str
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class SearchResult:
    """Single retrieval hit from the vector store."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


def retrieve_confluence_context(
    query_text: str,
    *,
    candidate_k: int = 10,
    final_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    model_name: str | None = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Retrieve a smaller, context-ready set of chunks for generation prep."""

    query_embedding, manifest = embed_confluence_query(
        query_text,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        encoder=encoder,
    )
    return retrieve_confluence_context_by_embedding(
        query_text,
        query_embedding,
        candidate_k=candidate_k,
        final_k=final_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=manifest.backend,
        min_score=min_score,
    )


def retrieve_confluence_context_by_embedding(
    query_text: str,
    query_embedding: list[float],
    *,
    candidate_k: int = 10,
    final_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    min_score: float | None = None,
) -> list[SearchResult]:
    """Retrieve, rerank, deduplicate, and trim search results for downstream use."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")
    if final_k <= 0:
        raise ValueError("final_k must be positive.")

    candidates = similarity_search_confluence_vector_store_by_embedding(
        query_embedding,
        top_k=max(candidate_k, final_k),
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
        min_score=min_score,
    )
    return _rerank_and_trim_results(
        query_text,
        candidates,
        final_k=final_k,
    )


def build_confluence_vector_store(
    input_dir: Path = CONFLUENCE_EMBEDDINGS_DIR,
    persist_dir: Path = VECTOR_DB_DIR,
    *,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
) -> VectorStoreBuildResult:
    """Build a persistent vector store from embedding-enriched Confluence chunk files."""

    records = _load_embedding_records(input_dir)
    manifest = _build_vector_store_from_records(
        records,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
    )

    return VectorStoreBuildResult(
        backend=manifest.backend,
        collection_name=manifest.collection_name,
        persist_dir=persist_dir,
        document_count=manifest.document_count,
        embedding_dimensions=manifest.embedding_dimensions,
        embedding_model=manifest.embedding_model,
        distance_metric=manifest.distance_metric,
    )


def query_confluence_vector_store(
    query_text: str,
    *,
    top_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    model_name: str | None = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
) -> list[SearchResult]:
    """Backward-compatible alias for similarity_search_confluence_vector_store."""

    return similarity_search_confluence_vector_store(
        query_text,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        encoder=encoder,
    )


def similarity_search_confluence_vector_store(
    query_text: str,
    *,
    top_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    model_name: str | None = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Run similarity search against the persistent Confluence vector store."""

    query_embedding, manifest = embed_confluence_query(
        query_text,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=device,
        encoder=encoder,
    )
    return similarity_search_confluence_vector_store_by_embedding(
        query_embedding,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=manifest.backend,
        min_score=min_score,
    )


def embed_confluence_query(
    query_text: str,
    *,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    model_name: str | None = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    device: str | None = None,
    encoder: TextEncoder | None = None,
) -> tuple[list[float], VectorStoreManifest]:
    """Embed a query string using the vector store's configured embedding model."""

    manifest = load_vector_store_manifest(
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
    )
    query_model = model_name or manifest.embedding_model or DEFAULT_EMBEDDING_MODEL
    active_encoder = encoder or load_sentence_transformer(model_name=query_model, device=device)
    vectors = encode_texts(
        active_encoder,
        [query_text],
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    return vectors[0], manifest


def search_confluence_vector_store_by_embedding(
    query_embedding: list[float],
    *,
    top_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
) -> list[SearchResult]:
    """Backward-compatible alias for similarity_search_confluence_vector_store_by_embedding."""

    return similarity_search_confluence_vector_store_by_embedding(
        query_embedding,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
    )


def similarity_search_confluence_vector_store_by_embedding(
    query_embedding: list[float],
    *,
    top_k: int = 5,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
    min_score: float | None = None,
) -> list[SearchResult]:
    """Search the persistent vector store using a precomputed query embedding."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    manifest = load_vector_store_manifest(
        persist_dir=persist_dir,
        collection_name=collection_name,
        backend=backend,
    )
    if len(query_embedding) != manifest.embedding_dimensions:
        raise ValueError(
            "Query embedding dimensions do not match the vector store: "
            f"{len(query_embedding)} != {manifest.embedding_dimensions}."
        )

    if manifest.backend == "chroma":
        results = _query_chroma_collection(
            query_embedding,
            top_k=top_k,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
    else:
        results = _query_faiss_index(
            query_embedding,
            top_k=top_k,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    return _filter_search_results(results, min_score=min_score)


def load_vector_store_manifest(
    *,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    backend: VectorBackend = "auto",
) -> VectorStoreManifest:
    """Read the manifest for a previously built vector store."""

    manifest_path = _manifest_path(persist_dir, collection_name)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No vector store manifest was found. "
            "Build the vector store before querying it."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = VectorStoreManifest(
        backend=cast(ResolvedBackend, payload["backend"]),
        collection_name=str(payload["collection_name"]),
        document_count=int(payload["document_count"]),
        embedding_dimensions=int(payload["embedding_dimensions"]),
        embedding_model=str(payload["embedding_model"]),
        distance_metric=str(payload.get("distance_metric", "cosine")),
    )
    if backend != "auto" and backend != manifest.backend:
        raise ValueError(
            f"Requested backend '{backend}' does not match built backend '{manifest.backend}'."
        )
    return manifest


def _build_vector_store_from_records(
    records: list[dict[str, Any]],
    *,
    persist_dir: Path,
    collection_name: str,
    backend: VectorBackend,
) -> VectorStoreManifest:
    if not records:
        raise ValueError("No embedding records were found in the input directory.")

    embedding_model, embedding_dimensions = _validate_embedding_records(records)
    resolved_backend = _resolve_backend(backend)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if resolved_backend == "chroma":
        _build_chroma_collection(records, persist_dir=persist_dir, collection_name=collection_name)
    else:
        _build_faiss_index(records, persist_dir=persist_dir, collection_name=collection_name)

    manifest = VectorStoreManifest(
        backend=resolved_backend,
        collection_name=collection_name,
        document_count=len(records),
        embedding_dimensions=embedding_dimensions,
        embedding_model=embedding_model,
    )
    _manifest_path(persist_dir, collection_name).write_text(
        json.dumps(asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_embedding_records(input_dir: Path) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Embedding directory does not exist: {input_dir}")

    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Embedding file contains a non-object record: {path}")
            records.append(payload)
    return records


def _validate_embedding_records(records: list[dict[str, Any]]) -> tuple[str, int]:
    first_record = records[0]
    embedding_model = str(first_record.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    first_embedding = _coerce_embedding(first_record.get("embedding"))
    embedding_dimensions = len(first_embedding)
    if embedding_dimensions == 0:
        raise ValueError("Embedding records must contain non-empty vectors.")

    for record in records:
        record_model = str(record.get("embedding_model") or embedding_model)
        if record_model != embedding_model:
            raise ValueError("All embedding records must use the same embedding model.")

        embedding = _coerce_embedding(record.get("embedding"))
        if len(embedding) != embedding_dimensions:
            raise ValueError("All embedding records must share the same dimensions.")

    return embedding_model, embedding_dimensions


def _resolve_backend(backend: VectorBackend) -> ResolvedBackend:
    if backend == "chroma":
        _require_chroma()
        return "chroma"
    if backend == "faiss":
        _require_faiss()
        return "faiss"

    try:
        _require_chroma()
        return "chroma"
    except RuntimeError:
        _require_faiss()
        return "faiss"


def _build_chroma_collection(
    records: list[dict[str, Any]],
    *,
    persist_dir: Path,
    collection_name: str,
) -> None:
    chromadb = _require_chroma()
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = max(1, int(client.get_max_batch_size()))
    for batch in _batched(records, batch_size):
        ids = [_record_id(record) for record in batch]
        embeddings = [_coerce_embedding(record.get("embedding")) for record in batch]
        documents = [str(record.get("text", "")) for record in batch]
        metadatas = [_coerce_chroma_metadata(record) for record in batch]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


def _query_chroma_collection(
    query_embedding: list[float],
    *,
    top_k: int,
    persist_dir: Path,
    collection_name: str,
) -> list[SearchResult]:
    chromadb = _require_chroma()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[SearchResult] = []
    for stored_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
        payload = _decode_payload(metadata, document)
        similarity = 1.0 - float(distance)
        results.append(
            SearchResult(
                chunk_id=str(payload.get("chunk_id", stored_id)),
                score=similarity,
                text=str(document or payload.get("text", "")),
                metadata=payload,
            )
        )

    return results


def _build_faiss_index(
    records: list[dict[str, Any]],
    *,
    persist_dir: Path,
    collection_name: str,
) -> None:
    faiss = _require_faiss()
    np = _require_numpy()

    vectors = [_normalize_vector(_coerce_embedding(record.get("embedding"))) for record in records]
    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(_faiss_index_path(persist_dir, collection_name)))

    stored_records = [_record_without_embedding(record) for record in records]
    lines = [json.dumps(record, ensure_ascii=False) for record in stored_records]
    _faiss_records_path(persist_dir, collection_name).write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _query_faiss_index(
    query_embedding: list[float],
    *,
    top_k: int,
    persist_dir: Path,
    collection_name: str,
) -> list[SearchResult]:
    faiss = _require_faiss()
    np = _require_numpy()

    index = faiss.read_index(str(_faiss_index_path(persist_dir, collection_name)))
    records = _load_faiss_records(_faiss_records_path(persist_dir, collection_name))
    query_vector = np.asarray([_normalize_vector(query_embedding)], dtype="float32")
    scores, row_indices = index.search(query_vector, top_k)

    results: list[SearchResult] = []
    for score, row_index in zip(scores[0].tolist(), row_indices[0].tolist(), strict=True):
        if row_index < 0:
            continue
        payload = records[row_index]
        results.append(
            SearchResult(
                chunk_id=str(payload.get("chunk_id", "")),
                score=float(score),
                text=str(payload.get("text", "")),
                metadata=payload,
            )
        )

    return results


def _coerce_embedding(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("Embedding record is missing an 'embedding' list.")
    if not value:
        raise ValueError("Embedding record contains an empty embedding.")
    return [float(item) for item in value]


def _coerce_chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    payload = json.dumps(_record_without_embedding(record), ensure_ascii=False)
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": str(record.get("chunk_id", "")),
        "payload_json": payload,
    }

    for key in ("page", "section", "space_key", "space_name", "source", "page_type", "source_path"):
        value = record.get(key)
        if value not in (None, ""):
            metadata[key] = str(value)

    word_count = record.get("word_count")
    if isinstance(word_count, int) and not isinstance(word_count, bool):
        metadata["word_count"] = word_count

    return metadata


def _decode_payload(metadata: object, document: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {"text": str(document or "")}

    payload_json = metadata.get("payload_json")
    payload: dict[str, Any]
    if isinstance(payload_json, str) and payload_json:
        decoded = json.loads(payload_json)
        payload = decoded if isinstance(decoded, dict) else {}
    else:
        payload = {}

    if "text" not in payload:
        payload["text"] = str(document or "")
    return payload


def _record_without_embedding(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "embedding"}


def _record_id(record: dict[str, Any]) -> str:
    chunk_id = str(record.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError("Embedding record is missing 'chunk_id'.")

    space_key = str(record.get("space_key", "")).strip()
    if space_key:
        return f"{space_key}::{chunk_id}"

    source_path = str(record.get("source_path", "")).strip()
    if source_path:
        return f"{source_path}::{chunk_id}"

    return chunk_id


def _load_faiss_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"FAISS record file contains a non-object record: {path}")
        records.append(payload)
    return records


def _rerank_and_trim_results(
    query_text: str,
    results: list[SearchResult],
    *,
    final_k: int,
) -> list[SearchResult]:
    query_keywords = _extract_query_keywords(query_text)
    page_counts = _count_pages(results)
    reranked = [
        _rerank_result(
            result,
            query_keywords=query_keywords,
            page_counts=page_counts,
        )
        for result in results
    ]
    reranked.sort(
        key=lambda result: (
            -result.score,
            -float(result.metadata.get("retrieval_similarity_score", result.score)),
            result.chunk_id,
        )
    )

    deduplicated: list[SearchResult] = []
    for result in reranked:
        if any(_is_near_duplicate(result, kept) for kept in deduplicated):
            continue
        deduplicated.append(result)
        if len(deduplicated) >= final_k:
            break

    return deduplicated


def _rerank_result(
    result: SearchResult,
    *,
    query_keywords: set[str],
    page_counts: dict[str, int],
) -> SearchResult:
    metadata = dict(result.metadata)
    page_title = _metadata_text(metadata, "page")
    section_title = _metadata_text(metadata, "section")

    page_key = _normalize_text(page_title)
    page_hit_count = page_counts.get(page_key, 0) if page_key else 0
    page_bonus = min(max(page_hit_count - 1, 0) * _PAGE_AGREEMENT_BOOST, _MAX_PAGE_AGREEMENT_BOOST)

    section_keywords = set(_tokenize_text(section_title))
    overlap = sorted(query_keywords & section_keywords)
    section_bonus = min(len(overlap) * _SECTION_KEYWORD_BOOST, _MAX_SECTION_KEYWORD_BOOST)

    reranked_score = result.score + page_bonus + section_bonus
    metadata["retrieval_similarity_score"] = result.score
    metadata["retrieval_rerank_score"] = reranked_score
    metadata["retrieval_page_hit_count"] = page_hit_count
    metadata["retrieval_section_keyword_overlap"] = overlap

    return SearchResult(
        chunk_id=result.chunk_id,
        score=reranked_score,
        text=result.text,
        metadata=metadata,
    )


def _count_pages(results: list[SearchResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        page_key = _normalize_text(_metadata_text(result.metadata, "page"))
        if not page_key:
            continue
        counts[page_key] = counts.get(page_key, 0) + 1
    return counts


def _is_near_duplicate(candidate: SearchResult, kept: SearchResult) -> bool:
    candidate_text = _normalize_text(candidate.text)
    kept_text = _normalize_text(kept.text)
    if candidate_text and candidate_text == kept_text:
        return True

    candidate_tokens = set(_tokenize_text(candidate.text))
    kept_tokens = set(_tokenize_text(kept.text))
    if not candidate_tokens or not kept_tokens:
        return False

    overlap_count = len(candidate_tokens & kept_tokens)
    if overlap_count < _MIN_OVERLAP_TOKENS:
        return False

    smaller_size = min(len(candidate_tokens), len(kept_tokens))
    if smaller_size == 0:
        return False

    return overlap_count / smaller_size >= _NEAR_DUPLICATE_OVERLAP_THRESHOLD


def _extract_query_keywords(query_text: str) -> set[str]:
    return {
        token
        for token in _tokenize_text(query_text)
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    }


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def _normalize_text(text: str) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    collapsed = _NORMALIZE_TEXT_PATTERN.sub(" ", ascii_text)
    return " ".join(collapsed.split())


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if value not in (None, "") else ""


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _filter_search_results(
    results: list[SearchResult],
    *,
    min_score: float | None,
) -> list[SearchResult]:
    if min_score is None:
        return results
    return [result for result in results if result.score >= min_score]


def _batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index:index + batch_size] for index in range(0, len(records), batch_size)]


def _manifest_path(persist_dir: Path, collection_name: str) -> Path:
    return persist_dir / f"{collection_name}.manifest.json"


def _faiss_index_path(persist_dir: Path, collection_name: str) -> Path:
    return persist_dir / f"{collection_name}.faiss"


def _faiss_records_path(persist_dir: Path, collection_name: str) -> Path:
    return persist_dir / f"{collection_name}.records.jsonl"


def _require_chroma() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Chroma is not installed. Install project dependencies or choose the FAISS backend."
        ) from exc
    return chromadb


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed. Install project dependencies or choose the Chroma backend."
        ) from exc
    return faiss


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("numpy is required for the FAISS backend.") from exc
    return numpy
