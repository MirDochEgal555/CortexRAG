# CortexRAG UI Usage

## Purpose

This guide explains how to run and use the current CortexRAG UI.

The UI is a local graph-first interface over the existing RAG pipeline. It depends on:

- a built vector store
- a persisted graph artifact
- the FastAPI backend
- the React frontend

For the full Zotero/Obsidian ingestion and build pipeline, see [zotero-obsidian-ingestion.md](./zotero-obsidian-ingestion.md).

## Prerequisites

Before using the UI, make sure the Python environment and frontend dependencies are installed.

Python setup from the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip install -e .
```

Frontend setup:

```powershell
cd frontend
npm install
```

The UI also needs a built retrieval corpus. If you have not built one yet, run the Zotero/Obsidian pipeline first, then build the vector store and graph:

```powershell
python scripts\preprocess_obsidian_vault.py
python scripts\preprocess_zotero_export.py
python scripts\chunk_obsidian_notes.py
python scripts\chunk_zotero_items.py
python scripts\embed_knowledge_chunks.py
python -m cortex_rag build-vector-store --with-graph
python -m cortex_rag verify-index
```

If the vector store already exists, rebuilding only the graph is enough:

```powershell
python -m cortex_rag build-graph
```

## Start The UI Stack

Run the backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src cortex_rag.api:create_app --factory --reload
```

Run the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Default local addresses:

- frontend: `http://127.0.0.1:5173`
- backend: `http://127.0.0.1:8000`

In local development, Vite proxies `/health`, `/search`, `/answer`, and `/graph/*` to the backend on port `8000`, so no extra frontend configuration is required for the default setup.

## Optional API Base URL

If the frontend should talk to a non-default backend URL, set `VITE_API_BASE_URL` before starting Vite.

Example:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

When `VITE_API_BASE_URL` is not set, the frontend uses relative API paths and relies on the Vite proxy.

## Using The Interface

### 1. Check backend readiness

When the page loads, the header checks `GET /health`.

- `cortex_rag backend ready` means the backend responded successfully
- `Backend unavailable` means the API could not be reached or failed during warmup

Warmup loads the persisted graph artifact and embedding model, so the first successful health check confirms the UI backend is usable.

### 2. Submit a query

Use the query box in the dock near the top of the screen.

- enter a question or retrieval prompt
- choose an answer mode
- click `Run query`

Current answer modes:

- `normal`
- `technical`
- `concise`
- `detailed`
- `bullet_summary`

A single submission triggers three backend calls:

- `/graph/neighborhood` for the graph slice
- `/answer` for the grounded answer
- `/search` for retrieval hits and scores

### 3. Read the graph canvas

The center panel renders the current knowledge neighborhood.

Node types:

- `document`: source page nodes
- `chunk`: retrieval-ready text units

Edge types:

- `belongs_to`: chunk-to-document membership
- `similar_to`: chunk-to-chunk similarity

Visual cues:

- highlighted nodes are direct retrieval hits
- query-path nodes and edges are emphasized
- non-path nodes and edges stay visible but de-emphasized
- document nodes use a rectangular shape
- chunk nodes use a circular shape

You can pan, zoom, and drag nodes directly in the canvas.

### 4. Use top hits for quick navigation

The query summary area shows the top retrieval hits.

Clicking a hit selects the corresponding chunk node in the graph and updates the detail panel context.

### 5. Read the grounded answer

The answer panel shows:

- the generated answer text
- answer mode
- backend
- total timing
- grounded source cards

Each source card includes:

- source label
- score
- page or section metadata
- source text snippet

If no grounded answer is available, the panel stays empty until a query succeeds.

### 6. Inspect node details

Click any node in the graph to populate the detail panel.

The detail panel includes:

- node summary
- page or source reference
- visible metadata fields
- retrieval evidence for that node
- related nodes
- edge explanations

This is the main explainability surface in the current UI.

### 7. Interpret retrieval evidence

The retrieval evidence section tells you whether the selected node was:

- directly retrieved by the search step
- included as part of the current query path
- outside the active query path

This helps distinguish direct evidence from supporting neighborhood context.

### 8. Interpret edge explanations

Edges carry readable explanations from graph metadata.

Current examples include:

- same document membership
- nearest-neighbor similarity
- shared metadata context when available

## Typical Workflow

1. Start the backend.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Confirm the backend status card is healthy.
5. Run a query.
6. Read the answer panel first.
7. Use the graph and detail panel to inspect why those sources were retrieved.

## Troubleshooting

### Backend unavailable

Likely causes:

- `uvicorn` is not running
- FastAPI is not installed in the active environment
- the backend failed to warm the embedding model or graph artifact

Check the backend terminal first.

### Health check fails even though the server is running

The `/health` endpoint warms runtime assets. It can fail if required local artifacts are missing.

Common causes:

- vector store manifest missing
- graph artifact missing
- embedding model configuration cannot be loaded

Rebuild the required local artifacts:

```powershell
python -m cortex_rag build-vector-store --with-graph
```

### Graph query returns an error

Common causes:

- the collection does not exist
- the graph file was not built for that collection
- retrieval returned an invalid local setup state

Rebuild the graph explicitly if needed:

```powershell
python -m cortex_rag build-graph
```

### Frontend loads but API calls fail

Check:

- the frontend is running on `127.0.0.1:5173`
- the backend is running on `127.0.0.1:8000`
- `VITE_API_BASE_URL` is either unset or points at the correct backend

### No meaningful answer appears

Common causes:

- retrieval returned weak or no matching chunks
- Ollama is unavailable
- the selected corpus does not contain the needed content

Validate the retrieval layer separately with:

```powershell
python -m cortex_rag similarity-search "your query here"
```

Validate grounded answering separately with:

```powershell
python -m cortex_rag ask "your query here" --mode technical
```

## Current Scope

The current UI is intentionally limited to:

- graph mode
- document and chunk nodes
- `belongs_to` and `similar_to` edges
- grounded answer inspection
- evidence-first explainability

It does not yet include chat mode, document explorer mode, entity/topic nodes, or animated retrieval effects.
