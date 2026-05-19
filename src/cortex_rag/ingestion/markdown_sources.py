"""Shared helpers for Markdown-based ingestion adapters."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

from cortex_rag.ingestion.confluence_chunks import (
    _build_page_chunks,
    _collapse_page_title_wrapper,
    _load_chunked_page,
    _parse_markdown_sections,
    ChunkedPage,
)


_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Parse the small YAML subset emitted by this project."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    metadata: dict[str, object] = {}
    current_key: str | None = None
    index = 1
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.strip() == "---":
            break
        if line.startswith("  - ") and current_key is not None:
            metadata.setdefault(current_key, [])
            assert isinstance(metadata[current_key], list)
            metadata[current_key].append(parse_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            current_key = None
            continue
        key, raw_value = line.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        current_key = key
        if raw_value:
            metadata[key] = parse_scalar(raw_value)
        else:
            metadata[key] = []

    return metadata, "\n".join(lines[index:]).strip()


def parse_scalar(value: str) -> object:
    """Parse a front-matter scalar from JSON-compatible YAML output."""

    if value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_markdown_document(metadata: dict[str, object], body: str) -> str:
    """Serialize metadata and body as a project-local Markdown document."""

    lines = ["---"]
    for key, value in metadata.items():
        lines.extend(_render_front_matter_item(key, value))
    lines.append("---")
    body = body.strip()
    if body:
        return "\n".join(lines) + "\n\n" + body + "\n"
    return "\n".join(lines) + "\n"


def _render_front_matter_item(key: str, value: object) -> list[str]:
    if value is None:
        return [f"{key}: null"]
    if isinstance(value, list):
        if not value:
            return [f"{key}: []"]
        lines = [f"{key}:"]
        lines.extend(f"  - {json.dumps(item, ensure_ascii=False)}" for item in value)
        return lines
    if isinstance(value, dict):
        return [f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"]
    return [f"{key}: {json.dumps(value, ensure_ascii=False)}"]


def first_markdown_heading(body: str) -> str | None:
    for line in body.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            return re.sub(r"[*_`]+", "", match.group(1)).strip()
    return None


def ensure_title_heading(body: str, title: str) -> str:
    if first_markdown_heading(body):
        return body.strip()
    return f"# {title}\n\n{body.strip()}".strip()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_ALNUM_PATTERN.sub("-", ascii_value).strip("-")


def path_stem_id(relative_path: Path) -> str:
    return relative_path.with_suffix("").as_posix().lower().replace(" ", "-")


def word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def chunk_markdown_file(
    path: Path,
    *,
    processed_root: Path,
    source: str,
    chunk_id_prefix: str,
    document_id: str,
    metadata_builder,
) -> list[dict[str, object]]:
    """Chunk a processed Markdown file and reshape records to the common contract."""

    page = _load_chunked_page(path)
    page_title = str(page.metadata.get("page_title") or page.metadata.get("title") or path.stem)
    root = _parse_markdown_sections(split_front_matter(path.read_text(encoding="utf-8"))[1], page_title=page_title)
    root = _collapse_page_title_wrapper(root, page_title=page_title)
    page = ChunkedPage(path=path, metadata={**page.metadata, "page_title": page_title}, root=root)

    raw_chunks = _build_page_chunks(page, page_index={}, processed_root=processed_root)
    output: list[dict[str, object]] = []
    for index, chunk in enumerate(raw_chunks, start=1):
        record_metadata = metadata_builder(page.metadata)
        chunk["chunk_id"] = f"{chunk_id_prefix}:{index:03d}"
        chunk["document_id"] = document_id
        chunk["source"] = source
        chunk["metadata"] = record_metadata
        chunk.pop("space_key", None)
        chunk.pop("space_name", None)
        chunk.pop("page_type", None)
        chunk.pop("source_html", None)
        chunk.pop("breadcrumbs", None)
        chunk.pop("created_by", None)
        chunk.pop("created_on", None)
        output.append(chunk)
    return output


def stable_output_name(title: str, fallback: str) -> str:
    return f"{slugify(title) or slugify(fallback) or 'item'}.md"
