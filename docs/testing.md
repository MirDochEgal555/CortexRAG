# Testing Guide

## Scope
The test suite covers the current Confluence ingestion, retrieval, and generation helpers. The most failure-prone edge cases around the local generation path are covered in:

- `tests/test_generation.py`
- `tests/test_cli.py`
- `tests/test_ask_confluence.py`

These tests specifically lock down:

- missing and empty prompt templates
- blank user questions rejected before prompt construction
- retrieved chunks with missing metadata
- empty retrieval results returned to the end-to-end ask script
- streaming Ollama responses that never yield any content
- CLI output when page and section metadata are absent
- timing output when no first-token latency is available

## Run Commands
Run the full suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run only the generation and CLI edge-case coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_generation.py tests\test_cli.py tests\test_ask_confluence.py
```

## Notes
- `pytest.ini` limits discovery to `tests/`.
- `tests/conftest.py` creates repo-local temporary directories under `scratch_pytest/`.
- The generation edge-case tests avoid live Ollama, SentenceTransformer, and vector-store dependencies by using fakes and monkeypatching.
