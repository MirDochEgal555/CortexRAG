# 🧠 CortexRAG UI Design
## Local Knowledge Graph Interface for Retrieval-Augmented Generation

---

## 🚀 Overview

This project implements a **knowledge graph-based UI** on top of a RAG system.

Instead of a traditional chatbot interface, the system visualizes knowledge as a **living graph**, where:

- **Nodes** represent knowledge units (documents, chunks, concepts)
- **Edges** represent relationships (semantic similarity, metadata, entities)

The result is a **“brain-like interface”** that allows users to:
- explore knowledge visually
- understand retrieval paths
- interact with data in an explainable way

---

## 🎯 Core Concept

> This is not just a RAG chatbot.

This is a:

## 👉 **Local Knowledge Operating System**

### Core components:
- Graph-based knowledge visualization
- Semantic retrieval engine
- Chat interface
- Document explorer

---

## 🎨 UI Vision

### 🧠 Main Graph Canvas (“Brain View”)

- Dark-mode interface
- Central interactive graph
- Force-directed layout

#### Nodes represent:
- documents
- chunks
- concepts
- topics
- entities (people, tools, projects)

#### Edges represent:
- semantic similarity
- shared metadata
- citations
- entity relationships
- user-defined links

---

## 🔍 Interactions

### Node Interaction
Clicking a node opens a detail panel with:

- summary
- source document
- related nodes
- top similar chunks
- suggested questions

---

### Query Interaction

User enters a query:

- relevant nodes are highlighted
- retrieval path is visualized
- graph “activates” like neural signals

---

### Visual Feedback

- highlighted nodes = relevant knowledge
- animated edges = retrieval flow
- clusters = semantic neighborhoods

---

## 🧩 UI Layout

### 1. Main Graph Canvas
- central visualization
- zoom, pan, drag
- clustering and grouping

---

### 2. Right-side Detail Panel

Displays:
- node title
- summary
- source reference
- related nodes
- similar chunks

---

### 3. Bottom Query Bar

Example queries:
- “What do I know about vector databases?”
- “Show me clusters around RAG”
- “Which notes connect AI automation and retrieval?”

---

### 4. Mode Switch

Switch between:
- Graph Mode
- Chat Mode
- Document Explorer
- Timeline Mode (optional)

---

## 🧠 Data Model

### Node Types
- documents
- chunks
- entities
- topics

### Edge Types
- `similar_to`
- `belongs_to`
- `mentions`
- `cites`
- `derived_from`
- `related_to`

---

## 🔗 Relationship Generation

### 1. Embedding Similarity
- connect nodes above similarity threshold

---

### 2. Shared Metadata
- same source
- same tags
- same topic

---

### 3. Entity Extraction
- connect nodes sharing:
  - tools
  - people
  - concepts
  - organizations

---

### 4. LLM-Inferred Links (Advanced)
- dependency relationships
- contradictions
- semantic grouping

---

## 🧱 Development Strategy

### ⚠️ Important
Do NOT build the full UI immediately.

---

## ✅ Phase 1 — MVP (RAG Core)

- ingest documents
- chunk data
- generate embeddings
- store in vector DB
- retrieve results
- simple text-based interface

---

## ✅ Phase 2 — Graph Mode

- nodes = documents / chunks
- edges = similarity / metadata
- interactive graph visualization
- node detail panel

---

## ✅ Phase 3 — Brain UI

- animated retrieval
- node highlighting
- clustering
- semantic neighborhoods
- query path visualization

---

## 🧪 Iterative UI Versions

### Version 1
- nodes = documents
- edges = similarity
- click → document view
- search highlights nodes

---

### Version 2
- nodes = chunks + topics
- edges = similarity + entities
- query highlights relevant chunks

---

### Version 3
- graph + chat combined
- answer panel includes:
  - used nodes
  - reasoning path
  - related knowledge

---

## 🏆 Key Advantages

Compared to standard RAG:

### Traditional:
- upload docs
- ask question
- get answer

---

### This system:
- visualizes knowledge
- explains retrieval
- shows relationships
- enables exploration

---

## 🎯 Benefits

- more intuitive understanding of data
- explainable AI behavior
- strong portfolio project
- extensible architecture
- engaging UX

---

## ⚙️ Tech Stack

### Backend
- Python + FastAPI
- ChromaDB (or FAISS)
- sentence-transformers
- Ollama (local LLM)

---

### Frontend
- React + TypeScript
- Tailwind CSS

---

### Graph Visualization
- Cytoscape.js (recommended)
- React Flow (alternative)
- Sigma.js (large graphs)
- D3.js (custom)

---

### Optional
- Neo4j (advanced graph DB)

---

## 💡 Killer Feature

> Ask a question and watch the graph light up.

- nodes activate based on relevance
- connections animate
- retrieval becomes visible

---

## 🧠 Learning Outcomes

This project teaches:

- vector databases
- embeddings
- chunking strategies
- retrieval systems
- graph modeling
- UI/UX for AI systems
- explainability

---

## 🏷️ Project Name

**CortexRAG**

---

## 🔥 Final Recommendation

Build this **layer by layer**:

1. RAG backend
2. simple graph
3. interactive UI
4. visual effects

---

## 🚫 Avoid

- overengineering early
- building full graph DB immediately
- complex animations before MVP
- huge datasets from the start

---

## 🚀 End Goal

A fully local, interactive, explainable AI system that feels like:

> **exploring your own brain**

---

## Next TODOs

The current repo is still in Phase 1: ingestion, embeddings, vector search, a package-level answer flow, and a thin API layer now exist, but there is still no frontend yet. The next work should stay narrow and produce a usable UI slice quickly.

### 1. Close out the current RAG surface

- Done: `src/cortex_rag/cli.py` now exposes a first-class `ask` command, and `scripts/ask_confluence.py` is only a thin compatibility wrapper.
- Done: the package now returns structured answer data from `src/cortex_rag/generation/confluence_answering.py`, including answer text, retrieved chunks, timings, and model metadata.
- Done: tests cover the package-level answer flow and CLI so a future web layer can call stable Python functions instead of shelling out to scripts.

### 2. Add a thin backend for the UI

- Done: a small FastAPI app now lives under `src/cortex_rag/api/`.
- Done: the first four endpoints exist: `/health`, `/search`, `/answer`, and `/graph/neighborhood`.
- Done: the API is read-only and wraps existing retrieval/generation code instead of inventing a separate graph database.
- Done: request/response schemas exist for search results, answer payloads, and graph nodes/edges.

### 3. Define the MVP graph model

- Start with only two node types: `document` and `chunk`.
- Start with only two edge types: `belongs_to` and `similar_to`.
- Build `document -> chunk` links from existing chunk metadata.
- Build `chunk <-> chunk` similarity edges from embedding search or an offline nearest-neighbor pass.
- Persist graph-ready JSON alongside the vector store so the frontend can render without expensive recomputation on every request.

### 4. Build Version 1 of the UI

- Scaffold a React + TypeScript frontend.
- Use Cytoscape.js first; it fits the current graph vision without forcing a custom rendering engine too early.
- Implement the initial layout from this doc:
  - center graph canvas
  - right detail panel
  - bottom query bar
- Keep the first visual pass simple: dark canvas, highlighted result nodes, basic pan/zoom/drag, no advanced animation yet.

### 5. Ship the first useful interaction loop

- Query submission calls `/search` and highlights the top matching document/chunk nodes.
- Clicking a node opens the detail panel with summary text, source page, section, and related nodes.
- The answer view calls `/answer` and shows both the response and the retrieved sources used to produce it.
- Add a mode toggle only after Graph Mode works; do not split effort across Chat Mode and Document Explorer yet.

### 6. Add explainability before polish

- Include retrieval scores and source metadata in the UI instead of hiding them.
- Render the query result path as a selected subgraph rather than animating the whole canvas.
- Capture why an edge exists: same document, nearest-neighbor similarity, or shared metadata.
- Defer "brain-like" animations until the static explainability layer is clear and debuggable.

### 7. Keep scope under control

- Do not introduce Neo4j yet.
- Do not model entities/topics until document/chunk nodes are stable.
- Do not build Timeline Mode in the first UI pass.
- Do not spend time on fancy motion until search, selection, and answer grounding are reliable.

### Suggested implementation order

1. Generate graph JSON for document/chunk nodes and edges.
2. Build the React graph shell and wire it to `/search`.
3. Add the detail panel and answer panel.
4. Add query-path highlighting and small visual polish.
