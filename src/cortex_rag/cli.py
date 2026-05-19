"""CLI entry points for CortexRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from cortex_rag.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_NUM_PREDICT,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_RAG_ANSWER_MODE,
    DEFAULT_RAG_PROMPT_PATH,
    DEFAULT_VECTOR_COLLECTION,
    VECTOR_DB_DIR,
)
from cortex_rag.generation import answer_confluence_question
from cortex_rag.graph import build_knowledge_graph
from cortex_rag.index_status import verify_current_searchable_index
from cortex_rag.retrieval import (
    SearchResult,
    build_knowledge_vector_store,
    retrieve_confluence_context,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="cortex_rag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-vector-store",
        help="Build or replace the persistent Zotero/Obsidian knowledge vector store.",
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
    build_parser.add_argument(
        "--with-graph",
        action="store_true",
        help="Also build the persisted document/chunk graph artifact alongside the vector store.",
    )
    build_parser.add_argument(
        "--graph-similarity-top-k",
        type=int,
        default=3,
        help="Number of similar chunk neighbors to persist per chunk when building the graph artifact.",
    )
    build_parser.add_argument(
        "--graph-similarity-threshold",
        type=float,
        default=0.6,
        help="Minimum cosine similarity required for a persisted chunk-to-chunk similarity edge.",
    )
    build_parser.set_defaults(handler=_run_build_vector_store)

    graph_parser = subparsers.add_parser(
        "build-graph",
        help="Build or replace the persisted Zotero/Obsidian document/chunk graph artifact for the UI backend.",
    )
    graph_parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Persistent collection name whose graph artifact should be built.",
    )
    graph_parser.add_argument(
        "--output-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the graph artifact should be persisted.",
    )
    graph_parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=3,
        help="Number of similar chunk neighbors to persist per chunk.",
    )
    graph_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Minimum cosine similarity required for a persisted chunk-to-chunk similarity edge.",
    )
    graph_parser.set_defaults(handler=_run_build_graph)

    verify_parser = subparsers.add_parser(
        "verify-index",
        help="Verify the current Zotero/Obsidian vector store and graph artifacts required by the UI.",
    )
    verify_parser.add_argument(
        "--backend",
        choices=("auto", "chroma", "faiss"),
        default="auto",
        help="Vector store backend. Defaults to the built backend recorded in the manifest.",
    )
    verify_parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Collection name to verify.",
    )
    verify_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the vector store and graph files are persisted.",
    )
    verify_parser.set_defaults(handler=_run_verify_index)

    search_parser = subparsers.add_parser(
        "similarity-search",
        help="Retrieve, rerank, deduplicate, and return context-ready knowledge chunks.",
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

    ask_parser = subparsers.add_parser(
        "ask",
        help="Retrieve Zotero/Obsidian context and generate a grounded answer with Ollama.",
    )
    ask_parser.add_argument("query", help="Question to answer from the knowledge vector store.")
    ask_parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of raw retrieval candidates before reranking.",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of reranked chunks to include in the prompt.",
    )
    ask_parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity score required for a chunk to be used.",
    )
    ask_parser.add_argument(
        "--backend",
        choices=("auto", "chroma", "faiss"),
        default="auto",
        help="Vector store backend. Defaults to the manifest backend when available.",
    )
    ask_parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Collection name to query.",
    )
    ask_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the vector store files are persisted.",
    )
    ask_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional SentenceTransformer model name or local path override for query embedding.",
    )
    ask_parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device override, for example cpu or cuda.",
    )
    ask_parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_RAG_PROMPT_PATH,
        help="Prompt template file used as the system message.",
    )
    ask_parser.add_argument(
        "--mode",
        choices=("concise", "normal", "detailed", "bullet_summary", "technical"),
        default=DEFAULT_RAG_ANSWER_MODE,
        help="Answer style injected into the prompt.",
    )
    ask_parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL.",
    )
    ask_parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name to use for answer generation.",
    )
    ask_parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_OLLAMA_TEMPERATURE,
        help="Generation temperature passed to Ollama.",
    )
    ask_parser.add_argument(
        "--num-ctx",
        type=int,
        default=min(DEFAULT_OLLAMA_NUM_CTX, 4096),
        help="Context window passed to Ollama.",
    )
    ask_parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_OLLAMA_NUM_PREDICT,
        help="Maximum number of tokens Ollama should generate.",
    )
    ask_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer as tokens arrive and report time to first token.",
    )
    ask_parser.set_defaults(handler=_run_ask)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CortexRAG CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


def _run_build_vector_store(args: argparse.Namespace) -> None:
    result = build_knowledge_vector_store(
        persist_dir=args.output_dir,
        collection_name=args.collection,
        backend=args.backend,
    )
    print(
        f"Built {result.backend} vector store '{result.collection_name}' "
        f"with {result.document_count} chunks at {result.persist_dir}."
    )
    if args.with_graph:
        graph_result = build_knowledge_graph(
            persist_dir=args.output_dir,
            collection_name=args.collection,
            similarity_top_k=args.graph_similarity_top_k,
            similarity_threshold=args.graph_similarity_threshold,
        )
        print(
            f"Built graph '{graph_result.collection_name}' with {graph_result.node_count} nodes "
            f"and {graph_result.edge_count} edges at {graph_result.persist_path}."
        )


def _run_build_graph(args: argparse.Namespace) -> None:
    result = build_knowledge_graph(
        persist_dir=args.output_dir,
        collection_name=args.collection,
        similarity_top_k=args.similarity_top_k,
        similarity_threshold=args.similarity_threshold,
    )
    print(
        f"Built graph '{result.collection_name}' with {result.node_count} nodes "
        f"and {result.edge_count} edges at {result.persist_path}."
    )


def _run_verify_index(args: argparse.Namespace) -> None:
    status = verify_current_searchable_index(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        backend=args.backend,
    )
    print(
        f"Verified {status.backend} searchable index '{status.collection_name}' "
        f"with {status.document_count} chunks at {status.persist_dir}."
    )
    print(f"Embedding model: {status.embedding_model}")
    print(f"Manifest: {status.manifest_path}")
    print(f"Graph: {status.graph_path}")
    print(f"Graph payload: {status.graph_node_count} nodes, {status.graph_edge_count} edges.")


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


def _run_ask(args: argparse.Namespace) -> None:
    if args.stream:
        print("Answer:")

    result = answer_confluence_question(
        args.query,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        min_score=args.min_score,
        backend=args.backend,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        device=args.device,
        prompt_path=args.prompt,
        answer_mode=args.mode,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        max_tokens=args.max_tokens,
        stream=args.stream,
        token_callback=_stream_token if args.stream else None,
    )

    if not result.sources:
        print("No relevant context was found. Ollama was not called.")
        print()
        _print_timings(result.timings)
        return

    if not args.stream:
        print("Answer:")
        print(result.answer)
        print()
    else:
        print()
        print()

    print("Sources:")
    _print_search_results(result.sources)
    print()
    _print_timings(result.timings)


def _print_search_results(results: list[SearchResult]) -> None:
    for index, result in enumerate(results, start=1):
        page = result.metadata.get("page", "")
        section = result.metadata.get("section", "")
        print(f"{index}. {result.chunk_id}  score={result.score:.4f}")
        if page or section:
            print(f"   {page} :: {section}".rstrip(" :"))
        print(f"   {result.text[:240].replace(chr(10), ' ')}")


def _print_timings(timings: object) -> None:
    print("Timings:")
    print(f"embedding: {_format_duration(getattr(timings, 'embedding_seconds'))}")
    print(f"retrieval: {_format_duration(getattr(timings, 'retrieval_seconds'))}")
    first_token_seconds = getattr(timings, "first_token_seconds")
    if first_token_seconds is not None:
        print(f"first_token: {_format_duration(first_token_seconds)}")
    print(f"generation: {_format_duration(getattr(timings, 'generation_seconds'))}")
    print(f"total: {_format_duration(getattr(timings, 'total_seconds'))}")


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _stream_token(token: str) -> None:
    print(token, end="", flush=True)


if __name__ == "__main__":
    main()
