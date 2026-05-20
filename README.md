# CortexRAG

Local Zotero/Obsidian-to-RAG pipeline with a graph-first UI, grounded answering, and a thin FastAPI backend.

The repository currently supports:

- Zotero and Obsidian preprocessing into Markdown, including matched Zotero PDF attachment text
- heading-aware chunking into retrieval records
- embedding generation with SentenceTransformers
- persistent vector-store build with Chroma or FAISS
- persisted document/chunk graph build for the UI
- retrieval and grounded answering through the package CLI
- a React + TypeScript frontend for graph exploration and answer inspection

## Docs

- Zotero/Obsidian workflow: [docs/zotero-obsidian-ingestion.md](docs/zotero-obsidian-ingestion.md)
- Confluence workflow: [docs/confluence-workflow.md](docs/confluence-workflow.md)
- Script order and runbook: [docs/confluence-script-order.md](docs/confluence-script-order.md)
- UI usage: [docs/ui-usage.md](docs/ui-usage.md)
- UI vision: [docs/ui-vision.md](docs/ui-vision.md)
- Testing: [docs/testing.md](docs/testing.md)
- Production deployment: [docs/production-deployment.md](docs/production-deployment.md)
- Server transfer runbook: [docs/server-transfer.md](docs/server-transfer.md)

## Quick Start

Create the virtual environment and install dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip install -e .
```

Put Obsidian vault copies under `data/raw/obsidian/` and a Zotero export under `data/raw/zotero/`, then run the indexing flow:

```powershell
python scripts\preprocess_obsidian_vault.py
python scripts\preprocess_zotero_export.py
python scripts\chunk_obsidian_notes.py
python scripts\chunk_zotero_items.py
python scripts\embed_knowledge_chunks.py
python -m cortex_rag build-vector-store --with-graph
python -m cortex_rag verify-index
```

That produces the main runtime artifacts:

- vector-store files under `storage/chroma/`
- a manifest at `storage/chroma/<collection>.manifest.json`
- a graph artifact at `storage/chroma/<collection>.graph.json`

## CLI Usage

The package CLI is now the primary query-time interface.

Inspect retrieval:

```powershell
python -m cortex_rag similarity-search "How are leads qualified?"
```

Ask a grounded question:

```powershell
python -m cortex_rag ask "What does the architecture say about the execution layer?"
python -m cortex_rag ask "How are leads qualified?" --mode technical
```

Scope retrieval to one Zotero paper:

```powershell
python -m cortex_rag ask-paper doe2024rag "What is the main contribution?"
python -m cortex_rag similarity-search "What dataset did they use?" --source zotero --citekey doe2024rag
```

Build only the graph artifact:

```powershell
python -m cortex_rag build-graph --similarity-top-k 3 --similarity-threshold 0.6
```

Compatibility note:

- `scripts\ask_confluence.py` still exists, but it is only a thin wrapper over `python -m cortex_rag ask ...`
- `scripts\build_confluence_graph.py` still exists, but it is only a thin wrapper over `python -m cortex_rag build-graph ...`

## UI Backend

The thin UI backend lives under `src/cortex_rag/api/` and exposes:

- `GET /health`
- `POST /search`
- `POST /answer`
- `POST /graph/neighborhood`

Run it locally from the repo root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src cortex_rag.api:create_app --factory --reload
```

The backend expects:

- a built vector store
- a built graph artifact
- access to the configured embedding model

## Frontend

The React + TypeScript frontend lives under `frontend/` and renders the current Brain View UI against the API backend.

Install and run it locally:

```powershell
cd frontend
npm install
npm run dev
```

Build it for production:

```powershell
cd frontend
npm run build
```

For the current frontend behavior and interaction model, see [docs/ui-usage.md](docs/ui-usage.md) and [docs/ui-vision.md](docs/ui-vision.md).

## Local UI Flow

Once the index exists, the normal local UI flow is:

1. start the backend
2. start the frontend
3. open `http://127.0.0.1:5173`
4. submit a query
5. inspect the graph neighborhood, grounded answer, and source evidence

## Testing

Run the full Python test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

There are currently no frontend automated tests. For UI changes, also run a frontend build and a manual smoke test. See [docs/testing.md](docs/testing.md) for the current testing strategy.

## Layout

```text
CortexRAG/
|-- data/
|   |-- raw/              # Source Zotero exports and Obsidian vault copies
|   |-- processed/        # Markdown pages
|   `-- chunks/           # Retrieval-ready chunk JSONL
|-- docs/                 # Project documentation
|-- frontend/             # React + TypeScript UI
|-- prompts/              # RAG prompt templates
|-- scripts/              # Indexing scripts and compatibility wrappers
|-- src/cortex_rag/       # Ingestion, retrieval, generation, graph, API, CLI
|-- storage/
|   |-- embeddings/       # Embedded chunk JSONL
|   `-- chroma/           # Vector-store and graph artifacts
`-- tests/
```

## Current Scope

The current shipped UI is intentionally narrow:

- graph mode only
- `document` and `chunk` nodes
- `belongs_to` and `similar_to` edges
- grounded answer inspection
- retrieval explainability before heavy visual polish

The near-term direction is to deepen the current graph workbench before expanding into additional modes or ontology layers.
