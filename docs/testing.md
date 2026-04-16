# Testing Guide

## Scope
The test suite covers the current Confluence ingestion, retrieval, and generation helpers. The most failure-prone edge cases around the local generation path are covered in:

- `tests/test_api_serializers.py`
- `tests/test_api_app.py`
- `tests/test_generation.py`
- `tests/test_confluence_answering.py`
- `tests/test_cli.py`
- `tests/test_ask_confluence.py`

These tests specifically lock down:

- missing and empty prompt templates
- blank user questions rejected before prompt construction
- retrieved chunks with missing metadata
- empty retrieval results returned to the package-level answer flow
- streaming Ollama responses that never yield any content
- CLI output when page and section metadata are absent
- timing output when no first-token latency is available
- API serialization for search, answer, and graph neighborhood payloads
- FastAPI route wiring for the thin UI backend when optional HTTP dependencies are installed
- the legacy ask script remaining a thin wrapper over the package CLI

## Run Commands
Run the full suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run only the generation and CLI edge-case coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_generation.py tests\test_confluence_answering.py tests\test_cli.py tests\test_ask_confluence.py
```

Run the API-focused coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_api_serializers.py tests\test_api_app.py
```

## Notes
- `pytest.ini` limits discovery to `tests/`.
- `tests/conftest.py` creates repo-local temporary directories under `scratch_pytest/`.
- The generation edge-case tests avoid live Ollama, SentenceTransformer, and vector-store dependencies by using fakes and monkeypatching.
