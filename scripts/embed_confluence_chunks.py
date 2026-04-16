"""Generate embeddings for Confluence chunk JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.config import DEFAULT_EMBEDDING_MODEL
from cortex_rag.retrieval import generate_confluence_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of chunks to encode per batch.")
    parser.add_argument("--device", default=None, help="Optional SentenceTransformer device override, for example cpu or cuda.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization on the generated embeddings.",
    )
    args = parser.parse_args()

    output_paths = generate_confluence_embeddings(
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
        device=args.device,
    )

    print(f"Wrote {len(output_paths)} embedding files.")
    for path in output_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
