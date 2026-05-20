"""Build and load an MVP document/chunk graph for Confluence content."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from cortex_rag.config import DEFAULT_VECTOR_COLLECTION, EMBEDDINGS_DIR, VECTOR_DB_DIR


GraphNodeType = Literal["document", "chunk"]
GraphEdgeType = Literal["belongs_to", "similar_to"]
CONFLUENCE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "confluence"
KNOWLEDGE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "knowledge"
SUMMARY_MAX_CHARS = 320


@dataclass(frozen=True)
class GraphNode:
    """Single graph node ready for JSON serialization."""

    id: str
    type: GraphNodeType
    label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphEdge:
    """Single graph edge ready for JSON serialization."""

    id: str
    source: str
    target: str
    type: GraphEdgeType
    weight: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphArtifact:
    """Persisted graph artifact consumed by the UI backend."""

    collection_name: str
    document_node_count: int
    chunk_node_count: int
    belongs_to_edge_count: int
    similar_to_edge_count: int
    similarity_top_k: int
    similarity_threshold: float
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class GraphBuildResult:
    """Summary of a completed graph build."""

    collection_name: str
    persist_path: Path
    document_node_count: int
    chunk_node_count: int
    belongs_to_edge_count: int
    similar_to_edge_count: int
    similarity_top_k: int
    similarity_threshold: float

    @property
    def node_count(self) -> int:
        return self.document_node_count + self.chunk_node_count

    @property
    def edge_count(self) -> int:
        return self.belongs_to_edge_count + self.similar_to_edge_count


@dataclass(frozen=True)
class GraphNeighborhood:
    """Subset of a graph relevant to one retrieval result set."""

    seed_node_ids: list[str]
    highlighted_node_ids: list[str]
    query_path_node_ids: list[str]
    query_path_edge_ids: list[str]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_confluence_graph(
    input_dir: Path = CONFLUENCE_EMBEDDINGS_DIR,
    persist_dir: Path = VECTOR_DB_DIR,
    *,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    similarity_top_k: int = 3,
    similarity_threshold: float = 0.6,
) -> GraphBuildResult:
    """Build and persist the MVP document/chunk graph from embedding records."""

    if similarity_top_k <= 0:
        raise ValueError("similarity_top_k must be positive.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0.")

    records = _load_embedding_records(input_dir)
    if not records:
        raise ValueError("No embedding records were found in the input directory.")

    document_nodes, chunk_nodes, belongs_to_edges = _build_membership_graph(records)
    similarity_edges = _build_similarity_edges(
        records,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
    )

    graph = GraphArtifact(
        collection_name=collection_name,
        document_node_count=len(document_nodes),
        chunk_node_count=len(chunk_nodes),
        belongs_to_edge_count=len(belongs_to_edges),
        similar_to_edge_count=len(similarity_edges),
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
        nodes=[*document_nodes.values(), *chunk_nodes.values()],
        edges=[*belongs_to_edges, *similarity_edges],
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    path = _graph_path(persist_dir, collection_name)
    payload = {
        "collection_name": graph.collection_name,
        "document_node_count": graph.document_node_count,
        "chunk_node_count": graph.chunk_node_count,
        "belongs_to_edge_count": graph.belongs_to_edge_count,
        "similar_to_edge_count": graph.similar_to_edge_count,
        "similarity_top_k": graph.similarity_top_k,
        "similarity_threshold": graph.similarity_threshold,
        "nodes": [asdict(node) for node in graph.nodes],
        "edges": [asdict(edge) for edge in graph.edges],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return GraphBuildResult(
        collection_name=collection_name,
        persist_path=path,
        document_node_count=graph.document_node_count,
        chunk_node_count=graph.chunk_node_count,
        belongs_to_edge_count=graph.belongs_to_edge_count,
        similar_to_edge_count=graph.similar_to_edge_count,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
    )


def build_knowledge_graph(
    input_dir: Path = KNOWLEDGE_EMBEDDINGS_DIR,
    persist_dir: Path = VECTOR_DB_DIR,
    *,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
    similarity_top_k: int = 3,
    similarity_threshold: float = 0.6,
) -> GraphBuildResult:
    """Build and persist the document/chunk graph from embedded Zotero and Obsidian chunks."""

    return build_confluence_graph(
        input_dir=input_dir,
        persist_dir=persist_dir,
        collection_name=collection_name,
        similarity_top_k=similarity_top_k,
        similarity_threshold=similarity_threshold,
    )


def load_confluence_graph(
    *,
    persist_dir: Path = VECTOR_DB_DIR,
    collection_name: str = DEFAULT_VECTOR_COLLECTION,
) -> GraphArtifact:
    """Load a previously persisted graph artifact."""

    path = _graph_path(persist_dir, collection_name)
    if not path.exists():
        raise FileNotFoundError(
            "No graph artifact was found. Build the graph before requesting graph neighborhoods."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    return GraphArtifact(
        collection_name=str(payload["collection_name"]),
        document_node_count=int(payload["document_node_count"]),
        chunk_node_count=int(payload["chunk_node_count"]),
        belongs_to_edge_count=int(payload["belongs_to_edge_count"]),
        similar_to_edge_count=int(payload["similar_to_edge_count"]),
        similarity_top_k=int(payload["similarity_top_k"]),
        similarity_threshold=float(payload["similarity_threshold"]),
        nodes=[
            GraphNode(
                id=str(node["id"]),
                type=node["type"],
                label=str(node["label"]),
                metadata=dict(node.get("metadata", {})),
            )
            for node in payload.get("nodes", [])
        ],
        edges=[
            GraphEdge(
                id=str(edge["id"]),
                source=str(edge["source"]),
                target=str(edge["target"]),
                type=edge["type"],
                weight=float(edge["weight"]) if edge.get("weight") is not None else None,
                metadata=dict(edge.get("metadata", {})),
            )
            for edge in payload.get("edges", [])
        ],
    )


def build_graph_neighborhood(
    graph: GraphArtifact,
    *,
    seed_chunk_ids: list[str],
    allowed_chunk_ids: list[str] | None = None,
) -> GraphNeighborhood:
    """Select a small neighborhood around retrieved chunk nodes."""

    nodes_by_id = {node.id: node for node in graph.nodes}
    allowed_node_ids = (
        {f"chunk::{chunk_id}" for chunk_id in allowed_chunk_ids}
        if allowed_chunk_ids is not None
        else None
    )
    edges_by_node: dict[str, list[GraphEdge]] = {}
    for edge in graph.edges:
        edges_by_node.setdefault(edge.source, []).append(edge)
        edges_by_node.setdefault(edge.target, []).append(edge)

    seed_node_ids = [f"chunk::{chunk_id}" for chunk_id in seed_chunk_ids if f"chunk::{chunk_id}" in nodes_by_id]
    included_nodes: set[str] = set(seed_node_ids)
    highlighted_nodes: set[str] = set(seed_node_ids)
    query_path_nodes: set[str] = set(seed_node_ids)
    query_path_edges: set[str] = set()
    included_edges: dict[str, GraphEdge] = {}

    for seed_node_id in seed_node_ids:
        for edge in edges_by_node.get(seed_node_id, []):
            other_node_id = edge.target if edge.source == seed_node_id else edge.source
            if not _edge_allowed_for_neighborhood(edge, other_node_id, allowed_node_ids):
                continue
            included_edges[edge.id] = edge
            query_path_edges.add(edge.id)
            included_nodes.add(other_node_id)
            query_path_nodes.add(other_node_id)
            if edge.type == "belongs_to":
                highlighted_nodes.add(other_node_id)

    chunk_like_nodes = {node_id for node_id in included_nodes if node_id.startswith("chunk::")}
    for chunk_node_id in list(chunk_like_nodes):
        for edge in edges_by_node.get(chunk_node_id, []):
            if edge.type != "belongs_to":
                continue
            included_edges[edge.id] = edge
            included_nodes.add(edge.source)
            included_nodes.add(edge.target)
            if chunk_node_id in query_path_nodes:
                query_path_edges.add(edge.id)
                query_path_nodes.add(edge.source)
                query_path_nodes.add(edge.target)

    nodes = [nodes_by_id[node_id] for node_id in nodes_by_id if node_id in included_nodes]
    edges = [edge for edge_id, edge in included_edges.items() if edge_id]

    return GraphNeighborhood(
        seed_node_ids=seed_node_ids,
        highlighted_node_ids=sorted(highlighted_nodes),
        query_path_node_ids=sorted(query_path_nodes),
        query_path_edge_ids=sorted(query_path_edges),
        nodes=nodes,
        edges=edges,
    )


def _edge_allowed_for_neighborhood(
    edge: GraphEdge,
    other_node_id: str,
    allowed_node_ids: set[str] | None,
) -> bool:
    if allowed_node_ids is None or edge.type == "belongs_to":
        return True
    if not other_node_id.startswith("chunk::"):
        return True
    return other_node_id in allowed_node_ids


def _build_membership_graph(
    records: list[dict[str, Any]],
) -> tuple[dict[str, GraphNode], dict[str, GraphNode], list[GraphEdge]]:
    document_nodes: dict[str, GraphNode] = {}
    chunk_nodes: dict[str, GraphNode] = {}
    belongs_to_edges: list[GraphEdge] = []
    document_chunk_counts: dict[str, int] = {}

    for record in records:
        chunk_id = str(record.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Embedding record is missing 'chunk_id'.")

        document_node = _document_node(record)
        chunk_node = _chunk_node(record)
        document_nodes.setdefault(document_node.id, document_node)
        chunk_nodes[chunk_node.id] = chunk_node
        document_chunk_counts[document_node.id] = document_chunk_counts.get(document_node.id, 0) + 1

        belongs_to_edges.append(
            GraphEdge(
                id=f"{document_node.id}--{chunk_node.id}::belongs_to",
                source=document_node.id,
                target=chunk_node.id,
                type="belongs_to",
                weight=1.0,
                metadata={
                    "reason": "same_document",
                    "explanation": "Same document: this chunk belongs to the source page represented by the document node.",
                },
            )
        )

    for document_id, node in list(document_nodes.items()):
        metadata = dict(node.metadata)
        metadata["chunk_count"] = document_chunk_counts.get(document_id, 0)
        document_nodes[document_id] = GraphNode(
            id=node.id,
            type=node.type,
            label=node.label,
            metadata=metadata,
        )

    return document_nodes, chunk_nodes, belongs_to_edges


def _build_similarity_edges(
    records: list[dict[str, Any]],
    *,
    similarity_top_k: int,
    similarity_threshold: float,
) -> list[GraphEdge]:
    chunk_vectors: list[tuple[str, str, list[float], dict[str, Any]]] = []
    for record in records:
        chunk_id = str(record.get("chunk_id", "")).strip()
        document_node_id = _document_node_id(record)
        embedding = _normalize_vector(_coerce_embedding(record.get("embedding")))
        chunk_vectors.append((f"chunk::{chunk_id}", document_node_id, embedding, record))

    edges_by_id: dict[str, GraphEdge] = {}
    for index, (chunk_node_id, document_node_id, vector, record) in enumerate(chunk_vectors):
        scored_neighbors: list[tuple[float, str, str, dict[str, Any]]] = []
        for other_index, (other_chunk_node_id, other_document_node_id, other_vector, other_record) in enumerate(chunk_vectors):
            if index == other_index:
                continue
            similarity = _dot(vector, other_vector)
            if similarity < similarity_threshold:
                continue
            scored_neighbors.append((similarity, other_chunk_node_id, other_document_node_id, other_record))

        scored_neighbors.sort(key=lambda item: (-item[0], item[1]))
        for rank, (similarity, other_chunk_node_id, other_document_node_id, other_record) in enumerate(
            scored_neighbors[:similarity_top_k],
            start=1,
        ):
            source, target = sorted([chunk_node_id, other_chunk_node_id])
            edge_id = f"{source}--{target}::similar_to"
            same_document = document_node_id == other_document_node_id
            shared_metadata = _shared_metadata_reasons(record, other_record)
            edges_by_id.setdefault(
                edge_id,
                GraphEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    type="similar_to",
                    weight=similarity,
                    metadata={
                        "reason": "nearest_neighbor_similarity",
                        "explanation": _similarity_explanation(
                            same_document=same_document,
                            shared_metadata=shared_metadata,
                        ),
                        "rank": rank,
                        "same_document": same_document,
                        "shared_metadata": shared_metadata,
                    },
                ),
            )

    return list(edges_by_id.values())


def _document_node(record: dict[str, Any]) -> GraphNode:
    return GraphNode(
        id=_document_node_id(record),
        type="document",
        label=_document_label(record),
        metadata={
            "page": _text(record.get("page")),
            "page_type": _text(record.get("page_type")),
            "source_path": _text(record.get("source_path")),
            "source_html": _text(record.get("source_html")),
            "source": _text(record.get("source")),
            "space_key": _text(record.get("space_key")),
            "space_name": _text(record.get("space_name")),
            "created_by": _text(record.get("created_by")),
            "created_on": _text(record.get("created_on")),
            "breadcrumbs": list(record.get("breadcrumbs", [])) if isinstance(record.get("breadcrumbs"), list) else [],
            "summary": _summarize_text(record.get("text")),
        },
    )


def _chunk_node(record: dict[str, Any]) -> GraphNode:
    chunk_id = str(record.get("chunk_id", "")).strip()
    label = _text(record.get("section")) or chunk_id
    return GraphNode(
        id=f"chunk::{chunk_id}",
        type="chunk",
        label=label,
        metadata={
            "chunk_id": chunk_id,
            "page": _text(record.get("page")),
            "section": _text(record.get("section")),
            "headings": list(record.get("headings", [])) if isinstance(record.get("headings"), list) else [],
            "source_path": _text(record.get("source_path")),
            "source": _text(record.get("source")),
            "space_key": _text(record.get("space_key")),
            "word_count": int(record["word_count"]) if isinstance(record.get("word_count"), int) else None,
            "summary": _summarize_text(record.get("text")),
        },
    )


def _document_node_id(record: dict[str, Any]) -> str:
    document_id = _text(record.get("document_id"))
    if document_id:
        return f"document::{document_id}"

    source_path = _text(record.get("source_path"))
    if source_path:
        return f"document::{source_path}"

    space_key = _text(record.get("space_key"))
    page = _text(record.get("page"))
    if space_key and page:
        return f"document::{space_key}::{page}"
    if page:
        return f"document::{page}"
    chunk_id = str(record.get("chunk_id", "")).strip()
    return f"document::{chunk_id.split(':', 1)[0]}"


def _document_label(record: dict[str, Any]) -> str:
    page = _text(record.get("page"))
    if page:
        return page
    source_path = _text(record.get("source_path"))
    if source_path:
        return Path(source_path).name
    return str(record.get("chunk_id", "")).split(":", 1)[0]


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


def _graph_path(persist_dir: Path, collection_name: str) -> Path:
    return persist_dir / f"{collection_name}.graph.json"


def _coerce_embedding(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("Embedding record is missing a non-empty 'embedding' list.")
    return [float(item) for item in value]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _summarize_text(value: object) -> str:
    text = " ".join(_text(value).split())
    if not text:
        return ""
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    truncated = text[: SUMMARY_MAX_CHARS - 3].rstrip()
    return f"{truncated}..."


def _shared_metadata_reasons(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    left_source = _text(left.get("source"))
    if left_source and left_source == _text(right.get("source")):
        reasons.append(f"same source ({left_source})")

    left_space_key = _text(left.get("space_key"))
    if left_space_key and left_space_key == _text(right.get("space_key")):
        reasons.append(f"same space ({left_space_key})")

    left_page_type = _text(left.get("page_type"))
    if left_page_type and left_page_type == _text(right.get("page_type")):
        reasons.append(f"same page type ({left_page_type})")

    return reasons


def _similarity_explanation(*, same_document: bool, shared_metadata: list[str]) -> str:
    parts = ["Nearest-neighbor similarity"]
    if same_document:
        parts.append("same document")
    if shared_metadata:
        parts.append(f"shared metadata: {', '.join(shared_metadata)}")
    return "; ".join(parts) + "."
