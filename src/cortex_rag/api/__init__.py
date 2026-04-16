"""Thin FastAPI backend for the CortexRAG UI."""

from cortex_rag.api.app import create_app

__all__ = ["create_app"]
