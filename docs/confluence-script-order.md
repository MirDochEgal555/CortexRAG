# Confluence Script Order

## Purpose

This is the operator runbook for the current Confluence pipeline. It answers one question only: what should be run, in what order, for indexing, querying, and the UI.

The important current distinction is:

- preprocessing, chunking, and embeddings are still script-first
- vector build, graph build, retrieval, and answer generation should prefer `python -m cortex_rag ...`

## Recommended Standard Order

Run the indexing flow from the repository root in this order:

1. `python scripts\preprocess_confluence_exports.py`
2. `python scripts\chunk_confluence_exports.py`
3. `python scripts\embed_confluence_chunks.py`
4. `python -m cortex_rag build-vector-store --with-graph`
5. `python -m cortex_rag verify-index`

After that, use one of these runtime paths:

1. `python -m cortex_rag similarity-search "<question>"`
2. `python -m cortex_rag ask "<question>"`
3. start the API backend and frontend if you want the browser UI

The first five steps refresh and verify the knowledge base. The later steps are query-time or UI-time commands.

## Indexing Commands

### 1. Preprocess raw Confluence exports

Run:

```powershell
python scripts\preprocess_confluence_exports.py
```

Reads:

- `data/raw/confluence/*.zip`

Writes:

- `data/processed/confluence/<SPACE_KEY>/*.md`

Run this when:

- you add a new Confluence HTML export zip
- an existing export zip changed

### 2. Chunk processed Markdown

Run:

```powershell
python scripts\chunk_confluence_exports.py
```

Reads:

- `data/processed/confluence/**/*.md`

Writes:

- `data/chunks/confluence/<SPACE_KEY>/*.jsonl`

Run this after:

- preprocessing

### 3. Generate embeddings

Run:

```powershell
python scripts\embed_confluence_chunks.py
```

Reads:

- `data/chunks/confluence/<SPACE_KEY>/*.jsonl`

Writes:

- `storage/embeddings/confluence/<SPACE_KEY>/*.jsonl`

Run this after:

- chunking

Useful variants:

```powershell
python scripts\embed_confluence_chunks.py --model sentence-transformers/all-mpnet-base-v2
python scripts\embed_confluence_chunks.py --batch-size 16 --device cpu
python scripts\embed_confluence_chunks.py --no-normalize
```

### 4. Build the persistent vector store and graph

Preferred run:

```powershell
python -m cortex_rag build-vector-store --with-graph
```

Reads:

- `storage/embeddings/confluence/**/*.jsonl`

Writes:

- `storage/chroma/<collection>.manifest.json`
- backend-specific vector-store files under `storage/chroma/`
- `storage/chroma/<collection>.graph.json`

Run this after:

- embeddings finish successfully

Useful variants:

```powershell
python -m cortex_rag build-vector-store --backend chroma
python -m cortex_rag build-vector-store --backend faiss
python -m cortex_rag build-vector-store --collection asa-dev
python -m cortex_rag build-vector-store --with-graph --graph-similarity-top-k 5 --graph-similarity-threshold 0.7
```

Legacy vector-build script still exists:

```powershell
python scripts\build_confluence_vector_store.py
```

If you use the legacy script, build the graph separately afterward.

### 5. Verify the searchable index

Use this before starting the UI backend or after copying artifacts to another machine:

```powershell
python -m cortex_rag verify-index
```

This checks the vector-store manifest, the backend-specific vector-store artifacts, and the graph artifact expected by `/graph/neighborhood`.

### 6. Build only the graph artifact

Use this when the vector store already exists and you only need to rebuild graph edges or graph metadata:

```powershell
python -m cortex_rag build-graph
```

Useful variants:

```powershell
python -m cortex_rag build-graph --collection asa-dev
python -m cortex_rag build-graph --similarity-top-k 5
python -m cortex_rag build-graph --similarity-threshold 0.7
```

Compatibility wrapper:

```powershell
python scripts\build_confluence_graph.py
```

## Query Commands

### 7. Inspect retrieval results

Preferred run:

```powershell
python -m cortex_rag similarity-search "What does the architecture say about the execution layer?"
```

Reads:

- the built vector store in `storage/chroma/`
- the embedding model defined in the manifest, unless overridden

Writes:

- nothing persistent

Use this when:

- you want to inspect retrieval quality before involving Ollama
- you want to tune `candidate-k`, `top-k`, or `min-score`

Useful variants:

```powershell
python -m cortex_rag similarity-search "How are leads qualified?" --candidate-k 10 --top-k 3
python -m cortex_rag similarity-search "How are leads qualified?" --min-score 0.7
python -m cortex_rag similarity-search "How are leads qualified?" --model C:\models\all-MiniLM-L6-v2
```

Legacy script:

```powershell
python scripts\query_confluence_vector_store.py "How are leads qualified?"
```

### 8. Ask a full RAG question

Preferred run:

```powershell
python -m cortex_rag ask "What does the architecture say about the execution layer?"
```

Reads:

- the built vector store in `storage/chroma/`
- `prompts/confluence_rag.md`
- Ollama runtime on the configured host

Writes:

- nothing persistent

Use this when:

- the vector store already exists
- you want a grounded answer instead of only retrieval results

Useful variants:

```powershell
python -m cortex_rag ask "How are leads qualified?" --mode technical
python -m cortex_rag ask "Summarize the architecture" --mode bullet_summary
python -m cortex_rag ask "What does the architecture say?" --top-k 2 --num-ctx 4096 --max-tokens 80
python -m cortex_rag ask "What does the architecture say?" --stream --max-tokens 80
python -m cortex_rag ask "How are leads qualified?" --ollama-model mistral:latest --temperature 0.1
```

Compatibility note:

- `scripts\ask_confluence.py` still works, but it forwards directly to `python -m cortex_rag ask ...`

## UI Commands

### 9. Start the API backend

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src cortex_rag.api:create_app --factory --reload
```

The backend expects:

- a built vector store manifest
- a built graph artifact
- access to the configured embedding model

Endpoints exposed:

- `GET /health`
- `POST /search`
- `POST /answer`
- `POST /graph/neighborhood`

### 10. Start the frontend

Run in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Default local URL:

- `http://127.0.0.1:5173`

The Vite dev server proxies API calls to `http://127.0.0.1:8000` by default.

See [ui-usage.md](./ui-usage.md) for the operator guide to the UI itself.

## Refresh Scenarios

### New or changed Confluence export

Run the full indexing flow again:

```powershell
python scripts\preprocess_confluence_exports.py
python scripts\chunk_confluence_exports.py
python scripts\embed_confluence_chunks.py
python -m cortex_rag build-vector-store --with-graph
```

### Embeddings already exist and you only need a rebuilt graph

Run:

```powershell
python -m cortex_rag build-graph
```

### No content change, just ask a question

Run only:

```powershell
python -m cortex_rag ask "<question>"
```

### Retrieval looks wrong and you want to debug before generation

Run:

```powershell
python -m cortex_rag similarity-search "<question>"
```

### You want the browser UI

Run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src cortex_rag.api:create_app --factory --reload
cd frontend
npm run dev
```

## Recommended First-Time Setup Flow

1. create and activate `.venv`
2. install `requirements-dev.txt`
3. place Confluence zip exports in `data/raw/confluence/`
4. run preprocess
5. run chunking
6. run embeddings
7. run `python -m cortex_rag build-vector-store --with-graph`
8. run a retrieval query
9. run `python -m cortex_rag ask ...` once Ollama is available
10. start the backend and frontend if you want the UI

## Common Mistakes

- Running `chunk_confluence_exports.py` before preprocessing. There will be no Markdown input to chunk.
- Running the vector-store build before embeddings exist. The build step will fail because there are no embedding records.
- Using the legacy vector-store script and forgetting to build the graph artifact afterward.
- Running the graph UI without building `confluence.graph.json`. `/graph/neighborhood` depends on the persisted graph artifact.
- Changing the embedding model and then querying an old index without rebuilding it. Query vectors must match the manifest dimensions and model assumptions.
- Changing embeddings or chunk boundaries and forgetting to rebuild the graph artifact. The graph should track the same chunk corpus as the vector store.
- Expecting `python -m cortex_rag ask ...` to work before the vector store exists.
- Expecting `python -m cortex_rag ask ...` to work if Ollama is not running or the configured model is missing.
- Starting the frontend without the backend. The page will load, but API-backed actions will fail.

## Minimal Daily Commands

To refresh the corpus:

```powershell
python scripts\preprocess_confluence_exports.py
python scripts\chunk_confluence_exports.py
python scripts\embed_confluence_chunks.py
python -m cortex_rag build-vector-store --with-graph
```

To inspect retrieval:

```powershell
python -m cortex_rag similarity-search "<question>"
```

To ask a question:

```powershell
python -m cortex_rag ask "<question>"
```

To run the UI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src cortex_rag.api:create_app --factory --reload
cd frontend
npm run dev
```
