"""Preprocess an Obsidian vault into normalized Markdown files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.obsidian import preprocess_obsidian_vault, preprocess_obsidian_vaults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, help="Path to one copied Obsidian vault.")
    args = parser.parse_args()

    if args.vault:
        output_paths = preprocess_obsidian_vault(args.vault)
    else:
        output_paths = preprocess_obsidian_vaults()

    print(f"Wrote {len(output_paths)} Markdown files.")
    for path in output_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
