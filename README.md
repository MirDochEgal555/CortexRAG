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

## Tools and Libraries
- Embeddings: Sentence Transformers
- Vector Store: FAISS (or similar)
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

## Project Structure
```text
CortexRAG/
├── data/
│   ├── raw/          # Source documents and exports
│   │   └── confluence/   # Confluence HTML exports (.zip), one per space
│   ├── processed/    # Cleaned text extracted from documents
│   │   └── confluence/   # Markdown pages generated from Confluence exports
│   └── chunks/       # Chunked text ready for embedding
├── notebooks/        # Experiments and one-off exploration
├── prompts/          # Prompt templates for local generation
├── scripts/          # Helper scripts for ingestion or maintenance
│   └── preprocess_confluence_exports.py
├── src/
│   └── cortex_rag/
│       ├── ingestion/   # Loading, preprocessing, and chunking
│       ├── retrieval/   # Embeddings and vector search
│       ├── generation/  # Local model interaction
│       ├── pipeline/    # End-to-end orchestration
│       ├── cli.py
│       └── config.py
├── storage/
│   └── chroma/       # Local vector database files
└── tests/            # Automated tests
```

## Generated Output Example
For an archive like `data/raw/confluence/ASA_2026-04-16.zip`, the script writes Markdown files such as:
- `data/processed/confluence/ASA/space-index.md`
- `data/processed/confluence/ASA/overview-3178688.md`
- `data/processed/confluence/ASA/architecture-3309569.md`
- `data/chunks/confluence/ASA/overview-3178688.jsonl`
- `data/chunks/confluence/ASA/architecture-3309569.jsonl`

## Implementation Notes
- The preprocessing logic lives in `src/cortex_rag/ingestion/confluence_html.py`.
- The chunking logic lives in `src/cortex_rag/ingestion/confluence_chunks.py`.
- The ingestion package exposes preprocessing and chunking entry points.
- The current converter uses only the Python standard library.

## Next Steps
1. Keep raw Confluence exports in `data/raw/confluence/`.
2. Re-run preprocessing whenever a new export is added.
3. Re-run chunk generation from `data/processed/confluence/`.
4. Build embeddings and retrieval on top of those chunks.
5. Connect the retrieval output to the local generation pipeline.
