"""Prompt-building helpers for Confluence-grounded generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cortex_rag.config import DEFAULT_RAG_ANSWER_MODE, DEFAULT_RAG_PROMPT_PATH
from cortex_rag.retrieval import SearchResult


AnswerMode = Literal["concise", "normal", "detailed", "bullet_summary", "technical"]
_ANSWER_MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    "concise": "Keep the answer short and direct, with only the essential points.",
    "normal": "Give a balanced answer with enough context to be useful without becoming long.",
    "detailed": "Give a thorough answer with supporting detail from the retrieved context.",
    "bullet_summary": "Answer as a compact bullet summary, focusing on the main takeaways.",
    "technical": "Use a technical style, emphasizing implementation details, structure, and precise terminology.",
}


def load_system_prompt(prompt_path: Path = DEFAULT_RAG_PROMPT_PATH) -> str:
    """Load the system prompt used for grounded answer generation."""

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template does not exist: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt template is empty: {prompt_path}")

    return prompt_text


def build_confluence_rag_messages(
    question: str,
    context_results: list[SearchResult],
    *,
    prompt_path: Path = DEFAULT_RAG_PROMPT_PATH,
    answer_mode: AnswerMode | str = DEFAULT_RAG_ANSWER_MODE,
) -> list[dict[str, str]]:
    """Build a system-plus-user message pair for RAG generation."""

    question_text = question.strip()
    if not question_text:
        raise ValueError("question must not be empty.")

    system_prompt = load_system_prompt(prompt_path)
    context_block = format_retrieval_context(context_results)
    normalized_mode = normalize_answer_mode(answer_mode)
    mode_instruction = _ANSWER_MODE_INSTRUCTIONS[normalized_mode]
    user_message = (
        f"Question:\n{question_text}\n\n"
        f"Answer mode: {normalized_mode}. {mode_instruction}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Write a grounded answer for the question using the retrieved context."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def format_retrieval_context(results: list[SearchResult]) -> str:
    """Render retrieved chunks into a plain-text context block."""

    if not results:
        return "No retrieved context was available."

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        page = _metadata_text(result.metadata, "page") or "Unknown page"
        section = _metadata_text(result.metadata, "section") or "Unspecified section"
        citation_metadata = _format_citation_metadata(result.metadata)
        metadata_lines = [f"Metadata: {citation_metadata}"] if citation_metadata else []
        blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Chunk ID: {result.chunk_id}",
                    f"Page: {page}",
                    f"Section: {section}",
                    f"Score: {result.score:.4f}",
                    *metadata_lines,
                    "Text:",
                    result.text.strip(),
                ]
            )
        )

    return "\n\n".join(blocks)


def normalize_answer_mode(answer_mode: AnswerMode | str) -> AnswerMode:
    """Validate and normalize the requested answer mode."""

    normalized = str(answer_mode).strip().lower()
    if normalized not in _ANSWER_MODE_INSTRUCTIONS:
        supported = ", ".join(_ANSWER_MODE_INSTRUCTIONS)
        raise ValueError(f"Unsupported answer mode '{answer_mode}'. Expected one of: {supported}.")
    return normalized  # type: ignore[return-value]


def _metadata_text(metadata: dict[str, object], key: str) -> str:
    value = _metadata_value(metadata, key)
    return str(value).strip() if value not in (None, "") else ""


def _format_citation_metadata(metadata: dict[str, object]) -> str:
    fields = [
        ("Authors", _metadata_value(metadata, "authors")),
        ("Year", _metadata_value(metadata, "year")),
        ("Publication", _metadata_value(metadata, "publication_title")),
        ("DOI", _metadata_value(metadata, "doi")),
        ("Citekey", _metadata_value(metadata, "citekey")),
        ("Tags", _metadata_value(metadata, "tags")),
    ]
    rendered = [
        f"{label}: {_format_metadata_value(value)}"
        for label, value in fields
        if value not in (None, "", [])
    ]
    return "; ".join(rendered)


def _metadata_value(metadata: dict[str, object], key: str) -> object:
    value = metadata.get(key)
    if value not in (None, ""):
        return value

    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        return nested_metadata.get(key)
    return None


def _format_metadata_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)
