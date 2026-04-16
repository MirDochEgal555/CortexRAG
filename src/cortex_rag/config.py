"""Central project paths and runtime defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

