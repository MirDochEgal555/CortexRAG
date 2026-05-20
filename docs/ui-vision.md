# CortexRAG UI Vision
## Local Knowledge Graph Interface for Retrieval-Augmented Generation

## Overview

CortexRAG is aiming at a local, explainable RAG interface where retrieval is visible instead of hidden behind a plain chat box.

The long-term direction is still a "brain view" for local knowledge, but the repository is no longer at the purely conceptual stage. The current codebase already includes a working Version 1 graph UI, a thin API backend, and a persisted graph artifact built from the Confluence pipeline.

This document describes both the product direction and the current implementation status.

## Product Direction

The project is moving toward a local knowledge operating system with four core capabilities:

- visual graph exploration of the knowledge base
- semantic retrieval over embedded Confluence chunks
- grounded answer generation
- inspectable evidence for why nodes and edges are shown

The important constraint remains the same: explainability comes before visual spectacle.

## Current Status

As of the current repo state, CortexRAG already ships the first usable graph workflow:

- a package CLI for building the vector store, building the graph, searching, and answering
- a persisted graph artifact at `storage/chroma/<collection>.graph.json`
- a FastAPI backend under `src/cortex_rag/api/`
- a React + TypeScript frontend under `frontend/`
- a Cytoscape-based graph canvas with answer and detail side panels

This means the project has moved past "Phase 1 only". The MVP graph mode exists and is queryable end to end.

## Implemented Today

### Backend and runtime

- `python -m cortex_rag ask` is the primary answer flow
- `scripts/ask_confluence.py` is now just a thin compatibility wrapper around the CLI
- the API exposes `GET /health`, `POST /search`, `POST /answer`, and `POST /graph/neighborhood`
- the backend warms the embedding model and persisted graph artifact before serving UI traffic
- answer responses include answer text, mode, backend, model, timings, and grounded sources

### Graph model

The current graph is intentionally narrow.

Node types:

- `document`
- `chunk`

Edge types:

- `belongs_to`
- `similar_to`

Current graph behavior:

- document nodes are derived from chunk source metadata
- chunk-to-document edges represent source membership
- chunk-to-chunk edges are generated offline from cosine similarity
- node metadata includes useful UI fields such as summaries, page/section labels, source path, headings, and counts
- edge metadata includes readable explanations such as same-document links and nearest-neighbor similarity context

### Frontend

The current UI is a real working shell, not a placeholder.

Implemented layout:

- hero/status bar
- query dock
- central graph canvas
- grounded answer panel
- right-side detail panel

Implemented interactions:

- submitting a query calls `/graph/neighborhood`, `/answer`, and `/search`
- result nodes are highlighted in the graph
- the query path is emphasized while non-path nodes and edges are visually de-emphasized
- clicking a node opens detail metadata, retrieval evidence, related nodes, and edge explanations
- the answer panel shows grounded sources with scores and source metadata
- top hits are exposed separately for quick navigation
- answer mode is selectable in the UI

Implemented visual direction:

- dark graph workspace
- static explainability cues instead of heavy animation
- pan/zoom/drag through Cytoscape
- responsive layout for desktop and smaller screens

## What Is Not Built Yet

The following ideas are still aspirational and should stay documented as future work rather than current capability:

- entity, concept, or topic nodes
- `mentions`, `cites`, `derived_from`, or user-authored edge types
- chat mode, document explorer mode, or timeline mode
- animated "neural signal" retrieval effects
- a separate graph database such as Neo4j
- write-back or graph editing workflows
- richer clustering, grouping, or semantic neighborhood controls

## Future Vision

The current Version 1 UI proves the base interaction loop. The next vision should not be "more panels" by default. It should be a deeper knowledge interface that still keeps retrieval legible.

The long-term product ambition is:

- asking questions against local knowledge
- seeing why the system answered the way it did
- exploring adjacent knowledge without losing context
- gradually turning retrieval into a navigable workspace instead of a single response

## Future Horizons

### Horizon 1: Better Graph Workbench

This is the nearest future and should grow directly out of the current UI.

Expected additions:

- graph controls for fit, reset, isolate selection, and depth expansion
- richer node filtering by page, space, node type, and score threshold
- better handling of dense neighborhoods through collapse, grouping, or lensing
- stronger visual hierarchy between seed hits, support context, and background nodes
- clearer transitions between one query state and the next

The purpose of this horizon is not new ontology. It is to make the existing document/chunk graph easier to read and operate.

### Horizon 2: Multi-Mode Knowledge Navigation

Once Graph Mode is stable, the UI can branch into distinct but connected modes.

Candidate modes:

- Graph Mode for neighborhood exploration
- Answer Mode for grounded response reading
- Document Mode for source-page inspection
- Search Mode for retrieval-first workflows
- Timeline Mode for date-heavy spaces, if the source corpus justifies it

The key requirement is shared state between modes:

- the same active query
- the same selected node
- the same source set
- the same evidence trail

The UI should feel like one knowledge session viewed through different lenses, not like separate apps glued together.

### Horizon 3: Richer Knowledge Model

The current graph uses only `document` and `chunk` nodes. A future graph can become semantically richer, but only when the additional structure improves navigation rather than creating visual noise.

Future node families:

- entities such as people, tools, teams, projects, and systems
- topics inferred from repeated chunk patterns
- tags or source-defined taxonomy nodes
- saved user collections or workspace nodes

Future edge families:

- `mentions`
- `related_to`
- `cites`
- `contradicts`
- `depends_on`
- user-authored links

This would shift the graph from "retrieval neighborhood" toward "knowledge map", but the bar for every new node or edge type should remain strict:

- it must improve explainability, discovery, or navigation
- it must not turn the canvas into unreadable background texture

### Horizon 4: Retrieval Explainability as a First-Class UI

The current explainability layer is already useful, but the future version should make the system's reasoning path inspectable at multiple levels.

Desired capabilities:

- query decomposition showing which terms or concepts drove retrieval
- rerank explanations showing why one chunk beat another
- prompt-construction inspection showing exactly which chunks were sent to the model
- answer-to-source alignment showing which source spans support which answer claims
- failure explanations when no grounded answer is produced

The best future UI does not only show the result path. It shows where uncertainty, omission, and fallback happened.

### Horizon 5: Personal Knowledge Workspace

The longer-term vision is bigger than a Confluence viewer.

CortexRAG can evolve into a local knowledge workspace where users:

- save queries and neighborhoods
- pin important nodes
- assemble temporary research boards
- compare multiple answers or retrieval paths
- annotate documents and chunks
- create explicit links that feed back into later exploration

At that point the graph is no longer only generated from source data. It also begins to reflect user intent and working context.

### Horizon 6: Agentic Research Surface

If the repo later grows beyond a pure retrieval UI, the graph can become an execution surface for agentic workflows.

Possible future interactions:

- ask the system to investigate a topic and open a live evidence board
- compare conflicting sources and cluster them by claim
- generate a synthesis draft while keeping every claim traceable to source chunks
- mark uncertain nodes for follow-up retrieval
- spawn reusable "knowledge trails" for recurring questions

This would only make sense if every agentic step remains inspectable. The graph should become the audit surface for agent behavior, not a decorative output.

## Future UX Principles

As the UI expands, these principles should stay fixed:

- explanation beats ornament
- a denser graph is not automatically a better graph
- every visual effect must communicate state, not just style
- every new mode must reuse the same evidence model
- every answer should remain reversible back to retrieval evidence
- local trust comes from inspectability, not from confident prose

## Future Visual Direction

The current dark graph shell is acceptable for Version 1, but the future visual language can become more intentional.

Potential directions:

- stronger visual distinction between retrieval state, answer state, and exploration state
- spatial memory cues so repeated visits to the same neighborhood feel familiar
- motion that communicates activation, dependency, or uncertainty
- layered background systems that help reveal cluster scale without overwhelming the graph
- selective use of animation for query-path activation, never as ambient noise

The "brain view" metaphor is still useful, but it should be interpreted carefully. The goal is not science-fiction chrome. The goal is a UI that makes knowledge feel alive, navigable, and accountable.

## Future Milestones

### Version 2

Likely focus:

- stronger graph controls
- better selection and navigation workflows
- larger-neighborhood readability
- early mode separation between graph exploration and document reading

### Version 3

Likely focus:

- additional node and edge families
- richer evidence views
- answer-to-source alignment
- saved sessions or pinned working sets

### Version 4+

Likely focus:

- collaborative or user-authored graph structure
- advanced explainability and audit tooling
- agentic research workflows
- a fuller local knowledge operating system rather than a single graph page

## Future Guardrails

Future vision should still respect a few limits:

- do not add ontology faster than the UI can explain it
- do not add animation faster than the product can justify it
- do not add modes that duplicate one another
- do not turn the graph into an unfiltered dump of every possible relationship
- do not sacrifice grounded evidence for a more dramatic interface

## Milestone Status

### 1. RAG core

Status: done

- ingestion, chunking, embeddings, vector storage, retrieval, and grounded answering exist
- the CLI is the stable package-level interface

### 2. Thin backend for the UI

Status: done

- FastAPI app exists under `src/cortex_rag/api/`
- request and response schemas exist for search, answers, and graph neighborhoods

### 3. MVP graph model

Status: done

- document and chunk nodes exist
- `belongs_to` and `similar_to` edges exist
- graph JSON is persisted beside the vector store

### 4. Version 1 UI

Status: done

- React + TypeScript frontend exists
- Cytoscape is integrated
- graph canvas, answer panel, and detail panel are live

### 5. First useful interaction loop

Status: done

- query -> retrieval -> graph neighborhood -> grounded answer works in one UI flow
- node selection and source inspection work

### 6. Explainability before polish

Status: largely done for Version 1

- retrieval scores are visible
- grounded sources are visible
- selected-node evidence is visible
- edge explanations are visible
- query-path emphasis is visible

What remains here is refinement, not invention.

### 7. Scope control

Status: still active

The original guardrails remain correct:

- do not add Neo4j yet
- do not add new node families until document/chunk behavior is stable
- do not split effort across multiple UI modes yet
- do not spend time on flashy animation before the retrieval story is solid

## Updated Tech Stack

### Backend

- Python
- FastAPI
- Chroma or FAISS
- sentence-transformers
- Ollama

### Frontend

- React 18
- TypeScript
- Vite
- hand-written CSS in `frontend/src/styles.css`

### Graph visualization

- Cytoscape.js

## Recommended Next Work

The next steps should be incremental and preserve the current narrow scope:

1. tighten the current graph UX with better loading and deeper selection controls
2. improve graph readability for larger neighborhoods before adding more node types
3. add small polish to empty/error states and evidence presentation
4. only then evaluate whether a second mode is justified

## Bottom Line

The repository now has a real Version 1 Brain View:

- local graph neighborhood retrieval
- grounded answering
- evidence inspection
- explainable node and edge rendering

The remaining work is to deepen this interface carefully, not to restart from a blank vision document.
