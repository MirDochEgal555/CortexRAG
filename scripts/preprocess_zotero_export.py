"""Preprocess a Zotero export into normalized Markdown files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.zotero import preprocess_zotero_export, preprocess_zotero_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to a Better BibTeX .bib or JSON export.")
    parser.add_argument("--notes-dir", type=Path, help="Optional notes/annotations directory.")
    parser.add_argument("--attachments-dir", type=Path, help="Optional attachment directory.")
    args = parser.parse_args()

    if args.input:
        output_paths = preprocess_zotero_export(
            args.input,
            notes_dir=args.notes_dir,
            attachments_dir=args.attachments_dir,
        )
    else:
        output_paths = preprocess_zotero_library()

    print(f"Wrote {len(output_paths)} Markdown files.")
    for path in output_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
