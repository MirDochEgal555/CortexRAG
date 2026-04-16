"""Query the persistent vector store with an embedded text prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.config import DEFAULT_VECTOR_COLLECTION, VECTOR_DB_DIR
from cortex_rag.retrieval import retrieve_confluence_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or search string to embed and retrieve against.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of raw embedding-similarity candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of reranked chunks to return after deduplication.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity score required for a result to be shown.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "chroma", "faiss"),
        default="auto",
        help="Vector store backend. Defaults to the built backend recorded in the manifest.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_VECTOR_COLLECTION,
        help="Collection name to query.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=VECTOR_DB_DIR,
        help="Directory where the vector store files are persisted.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device override, for example cpu or cuda.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional SentenceTransformer model name or local path override for query embedding.",
    )
    args = parser.parse_args()

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

    for index, result in enumerate(results, start=1):
        page = result.metadata.get("page", "")
        section = result.metadata.get("section", "")
        print(f"{index}. {result.chunk_id}  score={result.score:.4f}")
        if page or section:
            print(f"   {page} :: {section}".rstrip(" :"))
        print(f"   {result.text[:240].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
