# Lightweight Local RAG Q&A Chatbot

## Project Overview
This project is a lightweight Retrieval-Augmented Generation (RAG) system that answers questions from a small local knowledge base. The current ingestion workflow is built around Confluence space exports in HTML format, zipped per space, then normalized into Markdown for later chunking, embedding, and retrieval.

## Current Knowledge Ingestion Process

### 1. Export Confluence Spaces
- Export each Confluence space as an HTML export.
- Keep one `.zip` file per space.
- Store the archives in `data/raw/confluence/`.
- Prefer stable filenames such as `ASA_2026-04-16.zip`.

### 2. Preprocess HTML Exports into Markdown
- Run the preprocessing script:

```powershell
python scripts\preprocess_confluence_exports.py
```

- The script scans every zip in `data/raw/confluence/`.
- Each Confluence page is converted into a Markdown file under `data/processed/confluence/<SPACE_KEY>/`.
- The space landing page is also converted and written as `space-index.md`.

### 3. Preserve Useful Metadata
Every generated Markdown file includes YAML front matter with:
- `space_key`
- `space_name`
- `page_title`
- `page_type`
- `source_zip`
- `source_html`
- `breadcrumbs` when available
- `created_by` and `created_on` when present in the export

### 4. Normalize for Retrieval
During preprocessing, the converter:
- reads Confluence page content from the exported HTML structure
- strips export chrome such as styles and decorative images
- preserves headings, links, lists, blockquotes, code blocks, and tables
- resolves internal Confluence page links to local Markdown links
- produces stable Markdown filenames derived from page title and page id

### 5. Chunk Markdown into Retrieval Records
- Run the chunking script after preprocessing:

```powershell
python scripts\chunk_confluence_exports.py
```

- The script scans `data/processed/confluence/`.
- It writes JSONL chunk files under `data/chunks/confluence/<SPACE_KEY>/`.
- Chunks are heading-aware and target roughly `200-500` words when the source material is long enough.
- Each chunk includes page metadata plus internal page links when they can be resolved.

Example chunk:

```json
{
  "page": "RAG Architecture",
  "section": "Embeddings",
  "headings": ["RAG Architecture", "Embeddings"],
  "text": "Embeddings convert text into vectors...",
  "source": "confluence",
  "links": [
    {
      "text": "Retrieval",
      "target_path": "retrieval-222.md",
      "target_page": "Retrieval"
    }
  ]
}
```

### 6. Generate Embeddings from Chunk Files
- Run the embedding script after chunking:

```powershell
python scripts\embed_confluence_chunks.py
```

- The script scans `data/chunks/confluence/`.
- It writes embedding-enriched JSONL files under `storage/embeddings/confluence/<SPACE_KEY>/`.
- By default it uses `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings are L2-normalized by default, which makes cosine similarity and dot-product ranking equivalent.
- You can override the model, batch size, device, or normalization behavior:

```powershell
python scripts\embed_confluence_chunks.py --model sentence-transformers/all-mpnet-base-v2 --batch-size 16 --device cpu
python scripts\embed_confluence_chunks.py --no-normalize
```

- Each output record preserves the original chunk payload and appends `embedding_model`, `embedding_dimensions`, and `embedding`.

Example embedded chunk:

```json
{
  "chunk_id": "architecture-3309569:001",
  "page": "RAG Architecture",
  "section": "Embeddings",
  "text": "Embeddings convert text into vectors...",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding": [0.0142, -0.0371, 0.0924]
}
```

Notes:
- The first run may download the selected SentenceTransformer model unless it is already cached locally.
- If the machine is offline, pass a local model path to `--model`.

### 7. Build the Persistent Vector Store
- Run the vector-store build step after embeddings have been generated:

```powershell
python scripts\build_confluence_vector_store.py
```

- The script scans `storage/embeddings/confluence/`.
- It creates or replaces a persistent `confluence` collection under `storage/chroma/`.
- By default the project uses:
  - `chroma` on Python `3.13+`
  - `faiss` when Chroma is unavailable and FAISS is installed
- You can override the backend, collection name, or output directory:

```powershell
python scripts\build_confluence_vector_store.py --backend chroma
python scripts\build_confluence_vector_store.py --backend faiss --collection asa-dev
```

- A manifest file is written alongside the index so the query step can reuse the same backend and embedding dimensions:

```json
{
  "backend": "chroma",
  "collection_name": "confluence",
  "document_count": 57,
  "embedding_dimensions": 384,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "distance_metric": "cosine"
}
```

### 8. Retrieve Context Chunks
- Run the query script to retrieve context-ready chunks from the built store:

```powershell
python scripts\query_confluence_vector_store.py "What does the architecture say about the execution layer?"
```

- The query flow now uses a two-stage retrieval pass:
  - retrieve `10` raw embedding-similarity candidates by default
  - rerank them with page and section heuristics
  - drop near-duplicate chunks by normalized text and chunk overlap
  - keep the top `5` chunks by default, which you can lower to `3` when you want a tighter context window
- Context retrieval embeds the question with the same embedding model recorded in the manifest unless you override it.
- You can override the candidate pool size, final result count, or minimum similarity score:

```powershell
python scripts\query_confluence_vector_store.py "How are leads qualified?" --candidate-k 10 --top-k 3 --min-score 0.7
```

- If the model is not cached locally and the machine is offline, pass a local model path:

```powershell
python scripts\query_confluence_vector_store.py "How are leads qualified?" --model C:\models\all-MiniLM-L6-v2
```

- The reusable Python API exposes both high-level context retrieval and lower-level similarity search when you want to inspect or reuse the query vector:

```python
from pathlib import Path

from cortex_rag.retrieval import (
    embed_confluence_query,
    retrieve_confluence_context,
    retrieve_confluence_context_by_embedding,
    similarity_search_confluence_vector_store,
    similarity_search_confluence_vector_store_by_embedding,
)

context_results = retrieve_confluence_context(
    "How are leads qualified?",
    candidate_k=10,
    final_k=5,
    min_score=0.7,
    persist_dir=Path("storage/chroma"),
    collection_name="confluence",
)

query_embedding, manifest = embed_confluence_query(
    "How are leads qualified?",
    persist_dir=Path("storage/chroma"),
    collection_name="confluence",
)
context_results = retrieve_confluence_context_by_embedding(
    "How are leads qualified?",
    query_embedding,
    candidate_k=10,
    final_k=3,
    persist_dir=Path("storage/chroma"),
    collection_name=manifest.collection_name,
    backend=manifest.backend,
)

raw_results = similarity_search_confluence_vector_store_by_embedding(
    query_embedding,
    top_k=10,
    persist_dir=Path("storage/chroma"),
    collection_name=manifest.collection_name,
    backend=manifest.backend,
)
```

- The higher-level `retrieve_confluence_context(...)` path is intended for downstream generation prep.
- The lower-level `similarity_search_confluence_vector_store(...)` APIs remain available when you want raw nearest-neighbor results without reranking or deduplication.
- The older `query_confluence_vector_store(...)` and `search_confluence_vector_store_by_embedding(...)` names still work as compatibility aliases.

## Tools and Libraries
- Embeddings: Sentence Transformers
- Vector Store: Chroma or FAISS
- Local LLM: LLaMA/Mistral (via Ollama/LM Studio)

## Python Environment
This repository is set up to use a local virtual environment in `.venv`.

### Recommended Python Version
- Preferred: Python 3.11 or 3.12 for the widest package compatibility.
- Current local fallback: Python 3.13 works with the included dependency set by using `chromadb` instead of `faiss-cpu`.

### Create and Activate the Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
```

### Dependency Notes
- `faiss-cpu` installs automatically on Python versions below 3.13.
- `chromadb` is used automatically on Python 3.13 and newer as the vector store fallback.

## Testing
Run the test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Notes:
- If the virtual environment is already activated, `pytest -q` also works.
- Test discovery is limited to `tests/` via `pytest.ini`.
- Temporary test directories are provided by `tests/conftest.py` under `scratch_pytest/` to avoid Windows temp-directory permission issues seen on some machines.

## Project Structure
```text
CortexRAG/
|-- data/
|   |-- raw/              # Source documents and exports
|   |   `-- confluence/   # Confluence HTML exports (.zip), one per space
|   |-- processed/        # Cleaned text extracted from documents
|   |   `-- confluence/   # Markdown pages generated from Confluence exports
|   `-- chunks/           # Chunked text ready for embedding
|-- notebooks/            # Experiments and one-off exploration
|-- prompts/              # Prompt templates for local generation
|-- scripts/              # Helper scripts for ingestion or maintenance
|   |-- preprocess_confluence_exports.py
|   |-- chunk_confluence_exports.py
|   |-- embed_confluence_chunks.py
|   |-- build_confluence_vector_store.py
|   `-- query_confluence_vector_store.py
|-- src/
|   `-- cortex_rag/
|       |-- ingestion/    # Loading, preprocessing, and chunking
|       |-- retrieval/    # Embeddings and vector search
|       |-- generation/   # Local model interaction
|       |-- pipeline/     # End-to-end orchestration
|       |-- cli.py
|       `-- config.py
|-- storage/
|   |-- embeddings/       # Embedded chunk JSONL files
|   `-- chroma/           # Persistent Chroma or FAISS-backed vector-store files
`-- tests/                # Automated tests
```

## Generated Output Example
For an archive like `data/raw/confluence/ASA_2026-04-16.zip`, the script writes Markdown files such as:
- `data/processed/confluence/ASA/space-index.md`
- `data/processed/confluence/ASA/overview-3178688.md`
- `data/processed/confluence/ASA/architecture-3309569.md`
- `data/chunks/confluence/ASA/overview-3178688.jsonl`
- `data/chunks/confluence/ASA/architecture-3309569.jsonl`
- `storage/embeddings/confluence/ASA/overview-3178688.jsonl`
- `storage/embeddings/confluence/ASA/architecture-3309569.jsonl`
- `storage/chroma/confluence.manifest.json`
- `storage/chroma/chroma.sqlite3`

## Implementation Notes
- The preprocessing logic lives in `src/cortex_rag/ingestion/confluence_html.py`.
- The chunking logic lives in `src/cortex_rag/ingestion/confluence_chunks.py`.
- The embedding logic lives in `src/cortex_rag/retrieval/confluence_embeddings.py`.
- The vector-store build and query logic lives in `src/cortex_rag/retrieval/vector_store.py`.
- Similarity search is implemented as `similarity_search_confluence_vector_store(...)`.
- Context retrieval is implemented as `retrieve_confluence_context(...)`.
- Query embedding is implemented as `embed_confluence_query(...)` so the question-to-vector step can be reused independently from the search call.
- The ingestion and retrieval packages expose the current pipeline entry points.
- HTML preprocessing and chunking use only the Python standard library.
- Embedding generation and query embedding use `sentence-transformers`.
- Vector-store persistence uses Chroma when available and falls back to FAISS when requested.

## Next Steps
1. Keep raw Confluence exports in `data/raw/confluence/`.
2. Re-run preprocessing whenever a new export is added.
3. Re-run chunk generation from `data/processed/confluence/`.
4. Re-run embedding generation from `data/chunks/confluence/`.
5. Rebuild the vector store from `storage/embeddings/confluence/`.
6. Feed the reranked, deduplicated context results into the local generation pipeline.
