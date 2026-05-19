# Zotero and Obsidian Ingestion Workflow

## Purpose

This is the operator workflow for the current Zotero and Obsidian ingestion support in CortexRAG.

The implemented scope is ingest only:

- read copied/exported source files
- normalize them into Markdown under `data/processed/`
- chunk them into retrieval-ready JSONL under `data/chunks/`
- preserve source metadata for later embedding, vector-store, graph, and UI work

The ingestion code does not write back to Zotero, Obsidian, a live vault, or the Zotero SQLite database.

## Current Entry Points

Run these from the repository root.

Obsidian:

```bash
python scripts/preprocess_obsidian_vault.py --vault data/raw/obsidian/main
python scripts/chunk_obsidian_notes.py
```

Zotero:

```bash
python scripts/preprocess_zotero_export.py --input data/raw/zotero/library.bib
python scripts/chunk_zotero_items.py
```

The project requires Python `>=3.11`. If your local environment exposes that as `python3`, use `python3` in the commands.

## Data Flow

The implemented ingestion path is:

```text
data/raw/obsidian/<vault-name>/
  -> data/processed/obsidian/<vault-name>/
  -> data/chunks/obsidian/<vault-name>/

data/raw/zotero/library.bib
data/raw/zotero/notes/
data/raw/zotero/attachments/
  -> data/processed/zotero/
  -> data/chunks/zotero/
```

These outputs are source-parallel to the existing Confluence layout:

```text
data/
  raw/
    confluence/
    obsidian/
    zotero/
  processed/
    confluence/
    obsidian/
    zotero/
  chunks/
    confluence/
    obsidian/
    zotero/
```

## What Is Not Implemented Yet

The current Zotero and Obsidian work stops after chunk generation.

Not included yet:

- generic `scripts/embed_chunks.py`
- embedding Obsidian or Zotero chunks
- building source-specific Obsidian or Zotero vector collections
- building a mixed `knowledge` collection
- source filters in the UI
- Zotero PDF text extraction
- Zotero writeback
- Obsidian writeback

For now, the downstream Confluence embedding and vector-store pipeline remains unchanged.

## Chunk Record Contract

Both adapters emit JSONL records with the shared core fields expected by the future source-neutral pipeline:

```json
{
  "chunk_id": "zotero::doe2024rag:001",
  "document_id": "zotero::doe2024rag",
  "source": "zotero",
  "page": "Retrieval-Augmented Generation for Knowledge Work",
  "section": "Notes",
  "headings": [
    "Retrieval-Augmented Generation for Knowledge Work",
    "Notes"
  ],
  "text": "Chunk text used for embedding and retrieval.",
  "source_path": "retrieval-augmented-generation-for-knowledge-work.md",
  "word_count": 312,
  "links": [],
  "metadata": {
    "authors": ["Jane Doe"],
    "year": 2024,
    "doi": "10.xxxx/yyyy",
    "citekey": "doe2024rag",
    "tags": ["rag", "evaluation"]
  }
}
```

Required shared fields:

- `chunk_id`: globally unique stable chunk ID
- `document_id`: stable parent document or note ID
- `source`: `obsidian` or `zotero`
- `page`: human-readable title
- `section`: human-readable section title
- `headings`: Markdown heading path
- `text`: plain text used for later embedding
- `source_path`: processed source-relative path
- `word_count`: approximate chunk word count
- `links`: outbound links discovered during chunking
- `metadata`: source-specific preserved metadata

## Obsidian Workflow

### 1. Prepare Raw Input

Copy the vault, or a selected subset of the vault, into:

```text
data/raw/obsidian/<vault-name>/
```

Example:

```text
data/raw/obsidian/main/
  Projects/CortexRAG.md
  Literature/RAG Notes.md
```

Use a copied vault or exported subset when possible. The preprocessing step reads Markdown files and writes only to `data/processed/obsidian/`, but working from a copy keeps the source boundary obvious.

### 2. Preprocess the Vault

Run one vault explicitly:

```bash
python scripts/preprocess_obsidian_vault.py --vault data/raw/obsidian/main
```

Or preprocess every vault directory under `data/raw/obsidian/`:

```bash
python scripts/preprocess_obsidian_vault.py
```

Reads:

- `data/raw/obsidian/<vault-name>/**/*.md`

Writes:

- `data/processed/obsidian/<vault-name>/**/*.md`

The preprocessor skips hidden/plugin/cache-style directories, including `.obsidian`, `.git`, `.trash`, `node_modules`, `__pycache__`, `.cache`, `cache`, and `tmp`.

### 3. Obsidian Metadata Preserved

The processed Markdown front matter preserves:

- `source`
- `vault_name`
- `page_title`
- `source_path`
- front matter tags
- inline tags such as `#project/foo`
- wikilinks such as `[[Target Note]]`
- aliased wikilinks such as `[[Target Note|Label]]`
- Markdown links
- original front matter under `front_matter`

Title resolution order:

1. `title` from front matter
2. `page_title` from front matter
3. first Markdown heading
4. filename stem

Wikilinks are normalized in the readable Markdown body. For example, `[[Target Note|Label]]` becomes `Label` in the body, while the target and label are preserved in metadata.

### 4. Chunk Obsidian Notes

Run:

```bash
python scripts/chunk_obsidian_notes.py
```

Reads:

- `data/processed/obsidian/<vault-name>/**/*.md`

Writes:

- `data/chunks/obsidian/<vault-name>/**/*.jsonl`

Chunk IDs use this format:

```text
obsidian::<vault-name>::<normalized-note-path>:001
```

Example:

```text
obsidian::main::projects/cortexrag:001
```

Each chunk also gets:

```text
document_id = obsidian::<vault-name>::<normalized-note-path>
source = obsidian
```

## Zotero Workflow

### 1. Prepare Raw Input

Export the Zotero library to BibTeX or Better BibTeX, then place it under:

```text
data/raw/zotero/
```

Recommended layout:

```text
data/raw/zotero/
  library.bib
  notes/
  attachments/
```

Supported export inputs:

- `.bib`
- `.json`

The implementation intentionally does not read the live Zotero SQLite database.

### 2. Optional Notes and Attachments

Put exported notes or annotations under:

```text
data/raw/zotero/notes/
```

Put copied attachments under:

```text
data/raw/zotero/attachments/
```

Notes and attachments are matched by:

- Zotero item key
- citekey
- title slug

Matched note text is included in the normalized Markdown `## Notes` section. Matched attachment paths are preserved as metadata only. PDF text extraction is not part of the current ingestion implementation.

### 3. Preprocess the Zotero Export

Run with an explicit export:

```bash
python scripts/preprocess_zotero_export.py --input data/raw/zotero/library.bib
```

Optional explicit note and attachment directories:

```bash
python scripts/preprocess_zotero_export.py \
  --input data/raw/zotero/library.bib \
  --notes-dir data/raw/zotero/notes \
  --attachments-dir data/raw/zotero/attachments
```

Or preprocess the first `.bib` or `.json` export found under `data/raw/zotero/`:

```bash
python scripts/preprocess_zotero_export.py
```

Reads:

- `data/raw/zotero/*.bib` or `data/raw/zotero/*.json`
- optionally `data/raw/zotero/notes/`
- optionally `data/raw/zotero/attachments/`

Writes:

- `data/processed/zotero/*.md`

### 4. Zotero Metadata Preserved

The processed Markdown front matter preserves:

- `source`
- `zotero_key`
- `citekey`
- `page_title`
- `title`
- `authors`
- `year`
- `item_type`
- `publication_title`
- `doi`
- `isbn`
- `issn`
- `url`
- `abstract`
- `collections`
- `tags`
- `note_paths`
- `attachment_paths`
- `source_path`

The normalized Markdown body includes:

- `# <title>`
- `## Abstract`, when available
- `## Notes`, when matching note files are available
- `## Bibliography`, when citation metadata is available

### 5. Chunk Zotero Items

Run:

```bash
python scripts/chunk_zotero_items.py
```

Reads:

- `data/processed/zotero/*.md`

Writes:

- `data/chunks/zotero/*.jsonl`

Chunk IDs prefer citekey:

```text
zotero::<citekey>:001
```

Fallback:

```text
zotero::<zotero-key>:001
```

Each chunk also gets:

```text
document_id = zotero::<citekey-or-zotero-key>
source = zotero
```

## Full Ingestion Run

Use this when both source trees are ready:

```bash
python scripts/preprocess_obsidian_vault.py --vault data/raw/obsidian/main
python scripts/preprocess_zotero_export.py --input data/raw/zotero/library.bib

python scripts/chunk_obsidian_notes.py
python scripts/chunk_zotero_items.py
```

For multiple Obsidian vault directories under `data/raw/obsidian/`, omit `--vault`:

```bash
python scripts/preprocess_obsidian_vault.py
python scripts/chunk_obsidian_notes.py
```

## Validation Checklist

After preprocessing, check that normalized Markdown exists:

```bash
find data/processed/obsidian -name '*.md' | head
find data/processed/zotero -name '*.md' | head
```

After chunking, check that JSONL chunks exist:

```bash
find data/chunks/obsidian -name '*.jsonl' | head
find data/chunks/zotero -name '*.jsonl' | head
```

Inspect one chunk record:

```bash
head -n 1 data/chunks/zotero/<item>.jsonl
head -n 1 data/chunks/obsidian/<vault-name>/<note>.jsonl
```

Expected signs of a healthy chunk:

- `source` is `obsidian` or `zotero`
- `chunk_id` starts with the expected source prefix
- `document_id` is present
- `page`, `section`, and `headings` are populated
- `text` contains readable content
- `metadata` contains source-specific details

## Rebuild Rules

Run Obsidian preprocessing again when:

- copied vault files changed
- you add or remove Markdown notes
- front matter, tags, or links changed

Run Obsidian chunking again after:

- Obsidian preprocessing
- manual edits to `data/processed/obsidian/`

Run Zotero preprocessing again when:

- the Zotero export changed
- note exports changed
- attachment copies changed

Run Zotero chunking again after:

- Zotero preprocessing
- manual edits to `data/processed/zotero/`

Because the adapters are ingest-only, rebuilding is safe for the original sources. Rebuilds overwrite generated files under `data/processed/...` and `data/chunks/...`.

## Troubleshooting

If preprocessing writes zero files:

- confirm the raw input path exists
- confirm Obsidian files end in `.md`
- confirm the Zotero export ends in `.bib` or `.json`
- run from the repository root

If titles look wrong:

- for Obsidian, add `title` front matter or a top-level heading
- for Zotero, check that the export includes a `title` field

If Zotero notes are missing:

- confirm notes are under `data/raw/zotero/notes/`
- name note files with the citekey, Zotero item key, or title slug
- rerun preprocessing before chunking

If downstream search does not include Zotero or Obsidian:

- that is expected in the current implementation
- only ingestion and chunk generation are implemented for these sources
- generic embeddings and mixed-source vector-store support still need to be added
