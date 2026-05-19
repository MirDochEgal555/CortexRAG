"""Preprocess and chunk Zotero exports without writing back to Zotero."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from cortex_rag.config import CHUNKS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from cortex_rag.ingestion.markdown_sources import (
    build_markdown_document,
    chunk_markdown_file,
    slugify,
    stable_output_name,
)


ZOTERO_RAW_DIR = RAW_DATA_DIR / "zotero"
ZOTERO_PROCESSED_DIR = PROCESSED_DATA_DIR / "zotero"
ZOTERO_CHUNKS_DIR = CHUNKS_DIR / "zotero"

_BIB_ENTRY_START = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,", re.MULTILINE)
_FIELD_NAME = re.compile(r"\s*(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*=\s*")


@dataclass(slots=True)
class ZoteroItem:
    """Normalized Zotero item before Markdown serialization."""

    item_key: str
    item_type: str
    title: str
    fields: dict[str, object]

    @property
    def citekey(self) -> str:
        for key in ("citationkey", "citekey", "bibtexkey"):
            value = self.fields.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.item_key


def preprocess_zotero_export(
    input_path: Path,
    output_dir: Path = ZOTERO_PROCESSED_DIR,
    *,
    notes_dir: Path | None = None,
    attachments_dir: Path | None = None,
) -> list[Path]:
    """Normalize a Better BibTeX BibTeX or JSON export into Markdown files."""

    input_path = input_path.expanduser()
    if not input_path.exists():
        return []

    notes_dir = notes_dir or input_path.parent / "notes"
    attachments_dir = attachments_dir or input_path.parent / "attachments"

    if input_path.suffix.lower() == ".json":
        items = _parse_json_export(input_path)
    else:
        items = _parse_bibtex_export(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    seen_names: dict[str, int] = {}
    for item in items:
        metadata = _item_metadata(item, input_path=input_path, notes_dir=notes_dir, attachments_dir=attachments_dir)
        body = _item_body(item, metadata=metadata)
        output_name = _dedupe_name(stable_output_name(item.title, item.citekey), seen_names)
        output_path = output_dir / output_name
        output_path.write_text(build_markdown_document(metadata, body), encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def preprocess_zotero_library(
    input_dir: Path = ZOTERO_RAW_DIR,
    output_dir: Path = ZOTERO_PROCESSED_DIR,
) -> list[Path]:
    """Normalize the first Zotero export found under data/raw/zotero."""

    if not input_dir.exists():
        return []
    candidates = sorted([*input_dir.glob("*.bib"), *input_dir.glob("*.json")])
    if not candidates:
        return []
    return preprocess_zotero_export(candidates[0], output_dir)


def chunk_zotero_items(
    input_dir: Path = ZOTERO_PROCESSED_DIR,
    output_dir: Path = ZOTERO_CHUNKS_DIR,
) -> list[Path]:
    """Chunk processed Zotero item Markdown into retrieval-ready JSONL files."""

    if not input_dir.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for item_path in sorted(input_dir.glob("*.md")):
        metadata = _read_processed_metadata(item_path)
        item_key = str(metadata.get("citekey") or metadata.get("zotero_key") or item_path.stem)
        safe_key = slugify(item_key) or item_path.stem
        document_id = f"zotero::{safe_key}"
        chunks = chunk_markdown_file(
            item_path,
            processed_root=input_dir,
            source="zotero",
            chunk_id_prefix=document_id,
            document_id=document_id,
            metadata_builder=_zotero_metadata,
        )
        output_path = output_dir / f"{item_path.stem}.jsonl"
        lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def _parse_bibtex_export(path: Path) -> list[ZoteroItem]:
    text = path.read_text(encoding="utf-8")
    starts = list(_BIB_ENTRY_START.finditer(text))
    items: list[ZoteroItem] = []
    for index, match in enumerate(starts):
        body_start = match.end()
        body_end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[body_start:body_end].rsplit("}", maxsplit=1)[0]
        fields = _parse_bib_fields(body)
        item_key = match.group("key").strip()
        fields.setdefault("citationkey", item_key)
        title = str(fields.get("title") or item_key).strip("{} ")
        items.append(ZoteroItem(item_key=item_key, item_type=match.group("type"), title=title, fields=fields))
    return items


def _parse_bib_fields(body: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    position = 0
    while position < len(body):
        match = _FIELD_NAME.match(body, position)
        if not match:
            position += 1
            continue
        name = match.group("name").lower()
        value, position = _read_bib_value(body, match.end())
        fields[name] = _clean_bib_value(value)
        if position < len(body) and body[position] == ",":
            position += 1
    return fields


def _read_bib_value(body: str, position: int) -> tuple[str, int]:
    while position < len(body) and body[position].isspace():
        position += 1
    if position >= len(body):
        return "", position

    opener = body[position]
    if opener in {'"', "{"}:
        closer = '"' if opener == '"' else "}"
        depth = 0
        start = position + 1
        position += 1
        while position < len(body):
            char = body[position]
            if opener == "{" and char == "{":
                depth += 1
            elif char == closer:
                if depth == 0:
                    return body[start:position], position + 1
                depth -= 1
            position += 1
        return body[start:], position

    start = position
    while position < len(body) and body[position] not in ",\n":
        position += 1
    return body[start:position], position


def _clean_bib_value(value: str) -> str:
    value = value.replace("\\&", "&").replace("\\_", "_")
    value = re.sub(r"[{}]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_json_export(path: Path) -> list[ZoteroItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    items: list[ZoteroItem] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        fields = _flatten_json_item(record)
        item_key = str(fields.get("key") or fields.get("citationkey") or fields.get("citekey") or f"item-{index}")
        item_type = str(fields.get("itemtype") or fields.get("type") or "item")
        title = str(fields.get("title") or item_key)
        items.append(ZoteroItem(item_key=item_key, item_type=item_type, title=title, fields=fields))
    return items


def _flatten_json_item(record: dict[str, object]) -> dict[str, object]:
    data = record.get("data") if isinstance(record.get("data"), dict) else record
    assert isinstance(data, dict)
    fields: dict[str, object] = {str(key).lower(): value for key, value in record.items() if key != "data"}
    fields.update({str(key).lower(): value for key, value in data.items()})
    creators = data.get("creators")
    if isinstance(creators, list):
        fields["author"] = " and ".join(_creator_name(creator) for creator in creators if isinstance(creator, dict))
    tags = data.get("tags")
    if isinstance(tags, list):
        fields["keywords"] = ", ".join(
            str(tag.get("tag")) if isinstance(tag, dict) else str(tag) for tag in tags
        )
    return fields


def _creator_name(creator: dict[str, object]) -> str:
    if creator.get("name"):
        return str(creator["name"])
    return " ".join(str(creator.get(part, "")).strip() for part in ("firstName", "lastName")).strip()


def _item_metadata(
    item: ZoteroItem,
    *,
    input_path: Path,
    notes_dir: Path,
    attachments_dir: Path,
) -> dict[str, object]:
    authors = _split_authors(str(item.fields.get("author") or item.fields.get("authors") or ""))
    tags = _split_tags(str(item.fields.get("keywords") or item.fields.get("tags") or ""))
    notes = _find_related_files(notes_dir, item)
    attachments = _find_related_files(attachments_dir, item)
    return {
        "source": "zotero",
        "zotero_key": item.item_key,
        "citekey": item.citekey,
        "page_title": item.title,
        "title": item.title,
        "authors": authors,
        "year": _extract_year(str(item.fields.get("year") or item.fields.get("date") or "")),
        "item_type": item.item_type,
        "publication_title": item.fields.get("journaltitle") or item.fields.get("journal") or item.fields.get("booktitle"),
        "doi": item.fields.get("doi"),
        "isbn": item.fields.get("isbn"),
        "issn": item.fields.get("issn"),
        "url": item.fields.get("url"),
        "abstract": item.fields.get("abstract") or item.fields.get("abstractnote"),
        "collections": _split_tags(str(item.fields.get("collections") or "")),
        "tags": tags,
        "note_paths": [path.name for path in notes],
        "note_texts": [_read_text_file(path) for path in notes],
        "attachment_paths": [path.name for path in attachments],
        "source_path": input_path.name,
    }


def _item_body(item: ZoteroItem, *, metadata: dict[str, object]) -> str:
    parts = [f"# {item.title}"]
    abstract = metadata.get("abstract")
    if abstract:
        parts.extend(["", "## Abstract", "", str(abstract)])

    notes = [str(note).strip() for note in metadata.get("note_texts", []) if str(note).strip()]
    if notes:
        parts.extend(["", "## Notes", "", "\n\n".join(notes)])

    citation_bits = []
    if metadata.get("authors"):
        citation_bits.append(", ".join(str(author) for author in metadata["authors"]))
    if metadata.get("year"):
        citation_bits.append(str(metadata["year"]))
    if metadata.get("doi"):
        citation_bits.append(f"DOI: {metadata['doi']}")
    if metadata.get("url"):
        citation_bits.append(str(metadata["url"]))
    if citation_bits:
        parts.extend(["", "## Bibliography", "", "\n".join(f"- {bit}" for bit in citation_bits)])
    return "\n".join(parts)

def _find_related_files(directory: Path, item: ZoteroItem) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    keys = {item.item_key.lower(), item.citekey.lower(), slugify(item.title)}
    matches: list[Path] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        stem = path.stem.lower()
        if stem in keys or any(key and key in stem for key in keys):
            matches.append(path)
    return matches


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _split_authors(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+and\s+|;", value) if part.strip()]


def _split_tags(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]


def _extract_year(value: str) -> int | None:
    match = re.search(r"\b(1\d{3}|2\d{3})\b", value)
    return int(match.group(1)) if match else None


def _dedupe_name(name: str, seen: dict[str, int]) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix
    count = seen.get(name, 0)
    seen[name] = count + 1
    if not count:
        return name
    return f"{stem}-{count + 1}{suffix}"


def _read_processed_metadata(path: Path) -> dict[str, object]:
    from cortex_rag.ingestion.markdown_sources import split_front_matter

    metadata, _ = split_front_matter(path.read_text(encoding="utf-8"))
    return metadata


def _zotero_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "authors": metadata.get("authors", []),
        "year": metadata.get("year"),
        "doi": metadata.get("doi"),
        "isbn": metadata.get("isbn"),
        "issn": metadata.get("issn"),
        "url": metadata.get("url"),
        "citekey": metadata.get("citekey"),
        "zotero_key": metadata.get("zotero_key"),
        "item_type": metadata.get("item_type"),
        "publication_title": metadata.get("publication_title"),
        "collections": metadata.get("collections", []),
        "tags": metadata.get("tags", []),
        "note_paths": metadata.get("note_paths", []),
        "attachment_paths": metadata.get("attachment_paths", []),
    }
