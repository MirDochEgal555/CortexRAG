"""Chunk processed Confluence Markdown pages into retrieval-ready JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from cortex_rag.config import CHUNKS_DIR, PROCESSED_DATA_DIR


CONFLUENCE_PROCESSED_DIR = PROCESSED_DATA_DIR / "confluence"
CONFLUENCE_CHUNKS_DIR = CHUNKS_DIR / "confluence"

MIN_CHUNK_WORDS = 200
MAX_CHUNK_WORDS = 500
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class SectionNode:
    """A heading-scoped Markdown section."""

    title: str
    level: int
    path: list[str]
    content_lines: list[str] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


@dataclass(slots=True)
class ChunkPiece:
    """Intermediate chunk unit before serialization."""

    headings: list[str]
    raw_text: str
    word_count: int
    links: list[dict[str, str]]


@dataclass(slots=True)
class ChunkedPage:
    """Parsed processed Markdown page and its metadata."""

    path: Path
    metadata: dict[str, object]
    root: SectionNode

    @property
    def page_title(self) -> str:
        title = self.metadata.get("page_title")
        return str(title) if title else self.path.stem


def chunk_confluence_exports(
    input_dir: Path = CONFLUENCE_PROCESSED_DIR,
    output_dir: Path = CONFLUENCE_CHUNKS_DIR,
) -> list[Path]:
    """Chunk every processed Confluence page into JSONL files."""

    output_paths: list[Path] = []
    if not input_dir.exists():
        return output_paths

    for space_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        output_paths.extend(chunk_confluence_space(space_dir, input_dir=input_dir, output_dir=output_dir))
    return output_paths


def chunk_confluence_space(
    space_dir: Path,
    *,
    input_dir: Path = CONFLUENCE_PROCESSED_DIR,
    output_dir: Path = CONFLUENCE_CHUNKS_DIR,
) -> list[Path]:
    """Chunk all Markdown pages in a processed Confluence space directory."""

    markdown_paths = sorted(space_dir.glob("*.md"))
    if not markdown_paths:
        return []

    pages = [_load_chunked_page(path) for path in markdown_paths]
    page_index = {
        page.path.name: page.page_title
        for page in pages
    }

    space_output_dir = output_dir / space_dir.name
    space_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for page in pages:
        chunks = _build_page_chunks(page, page_index=page_index, processed_root=input_dir)
        output_path = space_output_dir / f"{page.path.stem}.jsonl"
        lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def _load_chunked_page(path: Path) -> ChunkedPage:
    metadata, body = _split_front_matter(path.read_text(encoding="utf-8"))
    root = _parse_markdown_sections(body, page_title=str(metadata.get("page_title") or path.stem))
    root = _collapse_page_title_wrapper(root, page_title=str(metadata.get("page_title") or path.stem))
    return ChunkedPage(path=path, metadata=metadata, root=root)


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
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
            metadata[current_key].append(_parse_front_matter_value(line[4:].strip()))
            continue
        if ":" not in line:
            current_key = None
            continue
        key, raw_value = line.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        current_key = key
        if raw_value:
            metadata[key] = _parse_front_matter_value(raw_value)
            continue
        metadata[key] = []

    body = "\n".join(lines[index:]).strip()
    return metadata, body


def _parse_front_matter_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_markdown_sections(body: str, *, page_title: str) -> SectionNode:
    root = SectionNode(title=page_title, level=0, path=[page_title])
    stack: list[SectionNode] = [root]

    for line in body.splitlines():
        heading_match = _HEADING_PATTERN.match(line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            title = _clean_heading_text(heading_match.group(2))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            section = SectionNode(title=title, level=level, path=parent.path + [title])
            parent.children.append(section)
            stack.append(section)
            continue
        stack[-1].content_lines.append(line.rstrip())

    return root


def _collapse_page_title_wrapper(root: SectionNode, *, page_title: str) -> SectionNode:
    if len(root.children) != 1:
        return root

    child = root.children[0]
    if _normalize_title(child.title) != _normalize_title(page_title):
        return root

    collapsed = SectionNode(
        title=page_title,
        level=0,
        path=[page_title],
        content_lines=[*root.content_lines, *child.content_lines],
        children=child.children,
    )
    _rebase_paths(collapsed)
    return collapsed


def _rebase_paths(node: SectionNode) -> None:
    for child in node.children:
        child.path = node.path + [child.title]
        _rebase_paths(child)


def _build_page_chunks(
    page: ChunkedPage,
    *,
    page_index: dict[str, str],
    processed_root: Path,
) -> list[dict[str, object]]:
    pieces = _chunk_root(page.root)
    chunks: list[dict[str, object]] = []
    page_relative_path = page.path.relative_to(processed_root).as_posix()

    for index, piece in enumerate(pieces, start=1):
        section = piece.headings[-1] if piece.headings else page.page_title
        chunks.append(
            {
                "chunk_id": f"{page.path.stem}:{index:03d}",
                "page": page.page_title,
                "section": section,
                "headings": piece.headings,
                "text": _flatten_markdown_text(piece.raw_text),
                "source": "confluence",
                "space_key": page.metadata.get("space_key"),
                "space_name": page.metadata.get("space_name"),
                "page_type": page.metadata.get("page_type"),
                "source_path": page_relative_path,
                "source_html": page.metadata.get("source_html"),
                "breadcrumbs": page.metadata.get("breadcrumbs", []),
                "created_by": page.metadata.get("created_by"),
                "created_on": page.metadata.get("created_on"),
                "word_count": piece.word_count,
                "links": _resolve_page_links(piece.links, page_index=page_index),
            }
        )

    return chunks


def _chunk_root(root: SectionNode) -> list[ChunkPiece]:
    if not root.children:
        return _chunk_node(root)

    child_pieces: list[ChunkPiece] = []
    for child in root.children:
        child_pieces.extend(_chunk_node(child))

    own_text = _normalize_block("\n".join(root.content_lines))
    if own_text:
        own_piece = _make_piece([root.path[0]], own_text)
        if child_pieces and own_piece.word_count < MIN_CHUNK_WORDS:
            combined_words = own_piece.word_count + child_pieces[0].word_count
            if combined_words <= MAX_CHUNK_WORDS:
                child_pieces[0] = _prepend_piece_text(own_piece.raw_text, child_pieces[0])
            else:
                child_pieces.insert(0, own_piece)
        else:
            child_pieces.insert(0, own_piece)

    return _merge_small_pieces(child_pieces)


def _chunk_node(node: SectionNode) -> list[ChunkPiece]:
    raw_text = _render_node_body(node)
    headings = node.path if node.level else [node.path[0]]
    word_count = _word_count(raw_text)
    if MIN_CHUNK_WORDS <= word_count <= MAX_CHUNK_WORDS:
        return [_make_piece(headings, raw_text)]

    if word_count > MAX_CHUNK_WORDS:
        if node.children:
            child_pieces: list[ChunkPiece] = []
            for child in node.children:
                child_pieces.extend(_chunk_node(child))

            own_text = _normalize_block("\n".join(node.content_lines))
            if own_text:
                own_piece = _make_piece(headings, own_text)
                if child_pieces and own_piece.word_count < MIN_CHUNK_WORDS:
                    combined_words = own_piece.word_count + child_pieces[0].word_count
                    if combined_words <= MAX_CHUNK_WORDS:
                        child_pieces[0] = _prepend_piece_text(own_piece.raw_text, child_pieces[0])
                    else:
                        child_pieces.insert(0, own_piece)
                else:
                    child_pieces.insert(0, own_piece)

            return _merge_small_pieces(child_pieces)

        return _split_large_leaf(headings, raw_text)

    return [_make_piece(headings, raw_text)]


def _render_node_body(node: SectionNode) -> str:
    parts: list[str] = []
    own_text = _normalize_block("\n".join(node.content_lines))
    if own_text:
        parts.append(own_text)
    for child in node.children:
        child_text = _render_descendant(child)
        if child_text:
            parts.append(child_text)
    return "\n\n".join(parts).strip()


def _render_descendant(node: SectionNode) -> str:
    parts: list[str] = [node.title]
    own_text = _normalize_block("\n".join(node.content_lines))
    if own_text:
        parts.append(own_text)
    for child in node.children:
        child_text = _render_descendant(child)
        if child_text:
            parts.append(child_text)
    return "\n\n".join(parts).strip()


def _make_piece(headings: list[str], raw_text: str) -> ChunkPiece:
    normalized_text = raw_text.strip()
    return ChunkPiece(
        headings=headings,
        raw_text=normalized_text,
        word_count=_word_count(normalized_text),
        links=_extract_links(normalized_text),
    )


def _prepend_piece_text(prefix_text: str, piece: ChunkPiece) -> ChunkPiece:
    merged_text = "\n\n".join(part for part in [prefix_text.strip(), piece.raw_text] if part).strip()
    return _make_piece(piece.headings, merged_text)


def _merge_small_pieces(pieces: list[ChunkPiece]) -> list[ChunkPiece]:
    if not pieces:
        return []

    merged: list[ChunkPiece] = []
    buffer: list[ChunkPiece] = []

    for piece in pieces:
        if not buffer:
            buffer = [piece]
            continue

        buffer_words = sum(item.word_count for item in buffer)
        should_merge = False
        if buffer_words < MIN_CHUNK_WORDS and buffer_words + piece.word_count <= MAX_CHUNK_WORDS:
            should_merge = True
        elif piece.word_count < MIN_CHUNK_WORDS and buffer_words + piece.word_count <= MAX_CHUNK_WORDS:
            should_merge = True

        if should_merge:
            buffer.append(piece)
            continue

        merged.append(_combine_pieces(buffer))
        buffer = [piece]

    if buffer:
        tail_piece = _combine_pieces(buffer)
        if (
            merged
            and tail_piece.word_count < MIN_CHUNK_WORDS
            and merged[-1].word_count + tail_piece.word_count <= MAX_CHUNK_WORDS
        ):
            merged[-1] = _combine_pieces([merged[-1], tail_piece])
        else:
            merged.append(tail_piece)

    return merged


def _combine_pieces(pieces: list[ChunkPiece]) -> ChunkPiece:
    if len(pieces) == 1:
        return pieces[0]

    headings = _common_heading_prefix([piece.headings for piece in pieces])
    rendered_parts: list[str] = []
    for piece in pieces:
        text = piece.raw_text
        if len(piece.headings) > len(headings):
            label = piece.headings[len(headings)]
            text = "\n\n".join(part for part in [label, text] if part).strip()
        if text:
            rendered_parts.append(text)
    merged_text = "\n\n".join(rendered_parts).strip()
    return _make_piece(headings or pieces[0].headings, merged_text)


def _split_large_leaf(headings: list[str], raw_text: str) -> list[ChunkPiece]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw_text) if part.strip()]
    if not paragraphs:
        return []

    pieces: list[ChunkPiece] = []
    current_parts: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = _word_count(paragraph)
        if paragraph_words > MAX_CHUNK_WORDS:
            if current_parts:
                pieces.append(_make_piece(headings, "\n\n".join(current_parts)))
                current_parts = []
                current_words = 0
            pieces.extend(_split_plain_text(headings, paragraph))
            continue

        if current_parts and current_words + paragraph_words > MAX_CHUNK_WORDS:
            pieces.append(_make_piece(headings, "\n\n".join(current_parts)))
            current_parts = [paragraph]
            current_words = paragraph_words
            continue

        current_parts.append(paragraph)
        current_words += paragraph_words

    if current_parts:
        pieces.append(_make_piece(headings, "\n\n".join(current_parts)))

    return _merge_small_pieces(pieces)


def _split_plain_text(headings: list[str], text: str) -> list[ChunkPiece]:
    words = _flatten_markdown_text(text).split()
    pieces: list[ChunkPiece] = []
    start = 0
    while start < len(words):
        end = min(start + MAX_CHUNK_WORDS, len(words))
        if end < len(words):
            target_end = min(start + 350, len(words))
            end = max(target_end, start + MIN_CHUNK_WORDS)
        pieces.append(_make_piece(headings, " ".join(words[start:end])))
        start = end
    return pieces


def _extract_links(text: str) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    links: list[dict[str, str]] = []

    for link_text, href in _LINK_PATTERN.findall(text):
        href = href.strip()
        if not href or "://" in href or href.startswith(("mailto:", "#")):
            continue

        target_path, _, anchor = href.partition("#")
        normalized_target = Path(target_path).name
        key = (link_text.strip(), normalized_target)
        if key in seen:
            continue
        seen.add(key)
        link_info = {
            "text": link_text.strip(),
            "target_path": normalized_target,
        }
        if anchor:
            link_info["anchor"] = anchor
        links.append(link_info)

    return links


def _resolve_page_links(
    links: list[dict[str, str]],
    *,
    page_index: dict[str, str],
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for link in links:
        link_payload = dict(link)
        target_path = link_payload.get("target_path")
        if target_path in page_index:
            link_payload["target_page"] = page_index[target_path]
        resolved.append(link_payload)
    return resolved


def _common_heading_prefix(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []

    prefix = list(paths[0])
    for path in paths[1:]:
        index = 0
        while index < len(prefix) and index < len(path) and prefix[index] == path[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _clean_heading_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _flatten_markdown_text(text: str) -> str:
    text = _LINK_PATTERN.sub(lambda match: match.group(1), text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text)
    text = re.sub(r"(?<!_)_(?!_)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_block(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(_flatten_markdown_text(text)))


def _normalize_title(text: str) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_ALNUM_PATTERN.sub("", ascii_text)
