"""Preprocess and chunk Obsidian Markdown vaults."""

from __future__ import annotations

import json
from pathlib import Path
import re

from cortex_rag.config import CHUNKS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from cortex_rag.ingestion.markdown_sources import (
    build_markdown_document,
    chunk_markdown_file,
    ensure_title_heading,
    first_markdown_heading,
    path_stem_id,
    split_front_matter,
)


OBSIDIAN_RAW_DIR = RAW_DATA_DIR / "obsidian"
OBSIDIAN_PROCESSED_DIR = PROCESSED_DATA_DIR / "obsidian"
OBSIDIAN_CHUNKS_DIR = CHUNKS_DIR / "obsidian"

_WIKILINK_PATTERN = re.compile(r"!\[\[([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_INLINE_TAG_PATTERN = re.compile(r"(?<![\w/])#([A-Za-z0-9_][A-Za-z0-9_/-]*)")
_SKIPPED_DIRS = {
    ".git",
    ".hg",
    ".obsidian",
    ".trash",
    "node_modules",
    "__pycache__",
    ".cache",
}


def preprocess_obsidian_vault(
    vault: Path,
    output_root: Path = OBSIDIAN_PROCESSED_DIR,
) -> list[Path]:
    """Normalize a copied Obsidian vault into processed Markdown files."""

    vault = vault.expanduser()
    if not vault.exists():
        return []
    if not vault.is_dir():
        raise ValueError(f"Obsidian vault must be a directory: {vault}")

    vault_name = vault.name
    output_dir = output_root / vault_name
    output_paths: list[Path] = []

    for source_path in sorted(vault.rglob("*.md")):
        if _should_skip(source_path, vault):
            continue

        relative_path = source_path.relative_to(vault)
        text = source_path.read_text(encoding="utf-8")
        original_front_matter, body = split_front_matter(text)
        normalized_body = _normalize_wikilinks(body)
        title = _resolve_title(original_front_matter, normalized_body, source_path)
        normalized_body = ensure_title_heading(normalized_body, title)

        wikilinks = _extract_wikilinks(body)
        metadata = {
            "source": "obsidian",
            "vault_name": vault_name,
            "page_title": title,
            "source_path": relative_path.as_posix(),
            "tags": _coerce_string_list(original_front_matter.get("tags")),
            "inline_tags": sorted(_extract_inline_tags(body)),
            "wikilinks": wikilinks,
            "markdown_links": _extract_markdown_links(body),
            "front_matter": original_front_matter,
        }

        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(build_markdown_document(metadata, normalized_body), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def preprocess_obsidian_vaults(
    input_root: Path = OBSIDIAN_RAW_DIR,
    output_root: Path = OBSIDIAN_PROCESSED_DIR,
) -> list[Path]:
    """Normalize every vault directory under data/raw/obsidian."""

    if not input_root.exists():
        return []
    output_paths: list[Path] = []
    for vault in sorted(path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        output_paths.extend(preprocess_obsidian_vault(vault, output_root))
    return output_paths


def chunk_obsidian_notes(
    input_root: Path = OBSIDIAN_PROCESSED_DIR,
    output_root: Path = OBSIDIAN_CHUNKS_DIR,
) -> list[Path]:
    """Chunk processed Obsidian notes into retrieval-ready JSONL files."""

    if not input_root.exists():
        return []

    output_paths: list[Path] = []
    for vault_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for note_path in sorted(vault_dir.rglob("*.md")):
            relative_path = note_path.relative_to(vault_dir)
            note_id = path_stem_id(relative_path)
            document_id = f"obsidian::{vault_dir.name}::{note_id}"
            chunks = chunk_markdown_file(
                note_path,
                processed_root=input_root,
                source="obsidian",
                chunk_id_prefix=document_id,
                document_id=document_id,
                metadata_builder=_obsidian_metadata,
            )
            output_path = output_root / vault_dir.name / relative_path.with_suffix(".jsonl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
            output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            output_paths.append(output_path)
    return output_paths


def _should_skip(path: Path, vault: Path) -> bool:
    relative_parts = path.relative_to(vault).parts
    return any(part.startswith(".") or part in _SKIPPED_DIRS or part.lower() in {"cache", "tmp"} for part in relative_parts)


def _resolve_title(metadata: dict[str, object], body: str, source_path: Path) -> str:
    for key in ("title", "page_title"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return first_markdown_heading(body) or source_path.stem


def _normalize_wikilinks(body: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1) or match.group(2) or ""
        target, _, label = raw.partition("|")
        text = label or Path(target.split("#", maxsplit=1)[0]).stem or target
        return text.strip()

    return _WIKILINK_PATTERN.sub(replace, body)


def _extract_wikilinks(body: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _WIKILINK_PATTERN.finditer(body):
        raw = match.group(1) or match.group(2) or ""
        target, _, label = raw.partition("|")
        note, _, heading = target.partition("#")
        item = {
            "target": note.strip(),
            "label": (label or Path(note).stem or note).strip(),
        }
        if heading:
            item["heading"] = heading.strip()
        key = (item["target"], item["label"], item.get("heading", ""))
        if key not in seen:
            seen.add(key)
            links.append(item)
    return links


def _extract_markdown_links(body: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for text, href in _MARKDOWN_LINK_PATTERN.findall(body):
        key = (text.strip(), href.strip())
        if key in seen:
            continue
        seen.add(key)
        links.append({"text": text.strip(), "href": href.strip()})
    return links


def _extract_inline_tags(body: str) -> set[str]:
    return {match.group(1) for match in _INLINE_TAG_PATTERN.finditer(body)}


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lstrip("#") for part in re.split(r"[,\s]+", value) if part.strip()]
    if isinstance(value, list):
        return [str(item).strip().lstrip("#") for item in value if str(item).strip()]
    return [str(value).strip().lstrip("#")]


def _obsidian_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "vault_name": metadata.get("vault_name"),
        "tags": metadata.get("tags", []),
        "inline_tags": metadata.get("inline_tags", []),
        "wikilinks": metadata.get("wikilinks", []),
        "markdown_links": metadata.get("markdown_links", []),
        "front_matter": metadata.get("front_matter", {}),
    }
