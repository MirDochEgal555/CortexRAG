"""CLI entry points for CortexRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from cortex_rag.config import DEFAULT_VECTOR_COLLECTION, VECTOR_DB_DIR
from cortex_rag.retrieval import (
    SearchResult,
    build_confluence_vector_store,
    retrieve_confluence_context,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="cortex_rag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-vector-store",
        help="Build or replace the persistent Confluence vector store.",
    )
    build_parser.add_argument(
        "--backend",
        choices=("auto", "chroma", "faiss"),
        default="auto",
        help="Vector store backend. Defaults to Chroma when available, otherwise FAISS.",
    )
    build_parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Persistent collection name to create or replace.",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the vector store files should be persisted.",
    )
    build_parser.set_defaults(handler=_run_build_vector_store)

    search_parser = subparsers.add_parser(
        "similarity-search",
        help="Retrieve, rerank, deduplicate, and return context-ready Confluence chunks.",
    )
    search_parser.add_argument(
        "query",
        help="Question or search string to embed and retrieve against.",
    )
    search_parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of raw embedding-similarity candidates to retrieve before reranking.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of reranked chunks to return after deduplication.",
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity score required for a result to be shown.",
    )
    search_parser.add_argument(
        "--backend",
        choices=("auto", "chroma", "faiss"),
        default="auto",
        help="Vector store backend. Defaults to the built backend recorded in the manifest.",
    )
    search_parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Collection name to query.",
    )
    search_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the vector store files are persisted.",
    )
    search_parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device override, for example cpu or cuda.",
    )
    search_parser.add_argument(
        "--model",
        default=None,
        help="Optional SentenceTransformer model name or local path override for query embedding.",
    )
    search_parser.set_defaults(handler=_run_similarity_search)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CortexRAG CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


def _run_build_vector_store(args: argparse.Namespace) -> None:
    result = build_confluence_vector_store(
        persist_dir=args.output_dir,
        collection_name=args.collection,
        backend=args.backend,
    )
    print(
        f"Built {result.backend} vector store '{result.collection_name}' "
        f"with {result.document_count} chunks at {result.persist_dir}."
    )


def _run_similarity_search(args: argparse.Namespace) -> None:
    results = retrieve_confluence_context(
        args.query,
        candidate_k=args.candidate_k,
        final_k=args.top_k,
        min_score=args.min_score,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        backend=args.backend,
        model_name=args.model,
        device=args.device,
    )
    _print_search_results(results)


def _print_search_results(results: list[SearchResult]) -> None:
    for index, result in enumerate(results, start=1):
        page = result.metadata.get("page", "")
        section = result.metadata.get("section", "")
        print(f"{index}. {result.chunk_id}  score={result.score:.4f}")
        if page or section:
            print(f"   {page} :: {section}".rstrip(" :"))
        print(f"   {result.text[:240].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
