"""Central project paths and runtime defaults."""

from __future__ import annotations

import os
from pathlib import Path


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
VECTOR_DB_DIR = CHROMA_DIR
EMBEDDINGS_DIR = STORAGE_DIR / "embeddings"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_COLLECTION = "knowledge"
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_NUM_CTX = _get_env_int("OLLAMA_NUM_CTX", 8192)
DEFAULT_OLLAMA_TEMPERATURE = _get_env_float("OLLAMA_TEMPERATURE", 0.2)
DEFAULT_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
DEFAULT_OLLAMA_NUM_PREDICT = _get_env_int("OLLAMA_NUM_PREDICT", 192)
DEFAULT_RAG_PROMPT_PATH = PROMPTS_DIR / "confluence_rag.md"
DEFAULT_RAG_ANSWER_MODE = os.getenv("RAG_ANSWER_MODE", "normal")
