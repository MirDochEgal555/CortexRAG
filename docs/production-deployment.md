# Production Deployment Guide

## Purpose

This document describes the cleanest production-style deployment for the current CortexRAG repository as it exists today.

The important qualifier is "as it exists today". This repo is already deployable, but its production path is shaped by current implementation choices:

- the backend is a thin FastAPI service under `src/cortex_rag/api/`
- the frontend is a separate Vite/React build under `frontend/`
- the RAG runtime depends on local persisted artifacts and a local Ollama service
- project data paths are currently resolved relative to the checked-out repository
- the backend does not currently configure CORS

This guide is therefore focused on a pragmatic first deployment, not an aspirational platform design.

## What Is Deployable Today

The repository already supports a production-style read-only query service with:

- Zotero/Obsidian preprocessing and indexing offline
- persistent vector-store artifacts
- persistent graph artifacts for the UI
- a FastAPI backend exposing `GET /health`, `POST /search`, `POST /answer`, and `POST /graph/neighborhood`
- a buildable frontend that calls those endpoints
- local Ollama-backed answer generation

The current production target should be:

- one Linux host
- one checked-out repo
- Docker Compose on one host
- one private Ollama service inside the Compose network
- one FastAPI backend container
- one web container serving the frontend and proxying API requests
- access restricted to VPN or another private network boundary

## Recommended Architecture

For the first stable deployment, keep the system on one machine with Docker Compose:

- web container: Nginx serving the built `frontend/dist`
- backend container: Uvicorn running `cortex_rag.api:create_app`
- model runtime: Ollama in the private Compose network
- persistent artifacts: repo-local `data/` and `storage/`

Request flow:

1. browser requests the frontend through the VPN-only web port
2. the web container proxies `/health`, `/search`, `/answer`, and `/graph/*` to the API container
3. the backend loads the vector-store manifest and graph artifact
4. the backend embeds the query and retrieves context
5. the backend optionally calls Ollama for grounded generation
6. the backend returns JSON to the frontend

## Same-Origin Requirement

The frontend can be configured with `VITE_API_BASE_URL`, but the backend currently does not enable CORS middleware.

That means the safest production setup is:

- serve the frontend and API from the same origin
- proxy API routes through the `web` container
- do not deploy the frontend on one origin and the backend on another unless you add CORS support first

For the current codebase, same-origin proxying is the correct default.

## Current Deployment Constraints

### 1. Deploy from a checked-out repo

`src/cortex_rag/config.py` resolves `PROJECT_ROOT` from the package file location and then derives:

- `data/`
- `storage/`
- `prompts/`

from that root.

That means the current code expects a real repo-style filesystem layout. Do not treat this as a wheel-only service yet.

Recommended deployment pattern:

- check out the repo on the server
- use an editable install with `pip install -e .`
- run the backend from that repo checkout

Until path configuration is refactored, this is the production-safe approach.

### 2. One backend worker first

The backend warms and caches the embedding model and graph artifact in-process.

If you run multiple workers:

- each worker loads its own model cache
- memory usage increases
- cold starts multiply

For the current repo, start with a single Uvicorn worker and scale only after measuring memory and latency.

### 3. The UI backend is read-only

The current API supports retrieval and answering only. It does not expose:

- authenticated admin endpoints
- upload endpoints
- background indexing endpoints
- write-back graph editing

Production deployment should assume offline artifact builds, then read-only serving.

## Recommended Production Layout

Example layout on Linux:

```text
/opt/cortexrag/app/            # checked-out repository
/opt/cortexrag/app/.env        # Compose binding settings, not secrets
```

Because the repo currently expects `data/` and `storage/` relative to the checkout, the simplest working layout is:

```text
/opt/cortexrag/app/data/
/opt/cortexrag/app/storage/
```

That is not the ideal long-term split, but it matches current code behavior.

## Runtime Dependencies

### Python

Current project metadata requires:

- Python `>=3.11`

Recommended server versions:

- Python `3.11`
- Python `3.12`

### Python packages

Current runtime dependencies in `requirements.txt` include:

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `ollama`
- `pypdf`
- `python-dotenv`
- `faiss-cpu` on Python versions below `3.13`
- `chromadb` on Python versions `>=3.13`

This split is unusual enough that you should pin and test the exact Python version you plan to run in production.

### Node

The frontend requires Node tooling only for the build step. You do not need Node at runtime if you serve the built static files from a reverse proxy.

## Ollama in Production

Ollama should run on the same host as the backend and remain bound to localhost.

Recommended initial production model:

- `llama3.2:3b`

Reason:

- it is already the default in `src/cortex_rag/config.py`
- it is the most conservative production choice in this repo today

Operational guidance:

- pin the model tag exactly
- pull the model during deployment, not on first live request
- do not expose Ollama directly to the public internet

Useful server checks:

```bash
ollama --version
ollama list
ollama run llama3.2:3b
```

The current code reads these environment-driven defaults:

```env
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=192
OLLAMA_TEMPERATURE=0.2
RAG_ANSWER_MODE=normal
```

## Artifact Strategy

The cleanest first production model is:

- build artifacts outside the live request path
- ship them to the server or build them during a controlled deployment step
- serve only from completed artifact sets

Required persisted artifacts:

- `storage/chroma/<collection>.manifest.json`
- vector-store backend files under `storage/chroma/`
- `storage/chroma/<collection>.graph.json`
- embedding files under `storage/embeddings/` if you expect to rebuild graph artifacts on the server

If the browser UI is deployed, the graph artifact is mandatory because `/graph/neighborhood` depends on it.

## Recommended Deployment Mode

### Best first production mode

- Linux host
- checked-out repo under `/opt/cortexrag/app`
- Docker Compose
- `web` container bound only to `127.0.0.1` or the server VPN IP
- `api` and `ollama` reachable only inside the Compose network
- repo-local `data/` and `storage/` mounted into the API container

### Defer for later

- cross-origin frontend/backend split
- multi-node inference or retrieval services
- online indexing from the public app

## Build and Ship Sequence

### 1. Build or refresh artifacts

From the repo root:

```bash
python scripts/preprocess_obsidian_vault.py
python scripts/preprocess_zotero_export.py
python scripts/chunk_obsidian_notes.py
python scripts/chunk_zotero_items.py
python scripts/embed_knowledge_chunks.py
python -m cortex_rag build-vector-store --with-graph
python -m cortex_rag verify-index
```

If production should stay read-only, do this in CI or in a staging build environment and ship the resulting `data/` and `storage/` contents with the deployment.

### 2. Build and start the containers

From the repo root on the server:

```bash
docker compose build
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b
```

The frontend build happens inside `Dockerfile.web`. The Python install happens inside `Dockerfile.api`.

## Backend Start Command

The API container runs:

```bash
python -m uvicorn --app-dir src cortex_rag.api:create_app --factory --host 0.0.0.0 --port 8000
```

Notes:

- the API port is not published on the host
- the `web` container is the only published service
- start with one worker
- use `GET /health` as the backend smoke test

The `/health` endpoint is meaningful here because it warms:

- the vector-store manifest
- the graph artifact
- the query embedding model

If `/health` fails, the deployment is not ready to serve traffic.

## Docker Web Proxy

The Docker web proxy does two jobs:

- serve the built frontend
- proxy API routes to the `api` service

The committed config is `deploy/nginx-docker.conf`. It aligns with the current frontend code, which calls:

- `/health`
- `/search`
- `/answer`
- `/graph/neighborhood`

without an `/api` prefix.

## Environment Management

The current repo uses environment variables primarily for Ollama and answer defaults. It does not yet read vector-store paths or collection names from environment variables in `src/cortex_rag/config.py`.

So for the current production deployment:

- use environment variables for Ollama and answer tuning
- keep the repo checked out in its expected layout
- keep production artifacts in the default repo-relative `data/` and `storage/` locations unless you also change request payloads or application code

Suggested runtime environment:

```env
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=192
OLLAMA_TEMPERATURE=0.2
RAG_ANSWER_MODE=normal
```

What is not yet truly environment-driven in the current code:

- `data/` location
- `storage/` location
- default collection name through startup config
- frontend/backend cross-origin behavior

The deployment doc should reflect that reality instead of pretending those knobs already exist.

## Security Baseline

Before exposing the service beyond a trusted local machine:

- expose only the `web` container port
- bind the published port to `127.0.0.1` or the server VPN IP
- keep Ollama unexposed inside Compose
- restrict who can reach the site through VPN membership and server firewall rules
- do not expose ingestion commands or raw artifact directories over HTTP

Important current gap:

- the backend has no authentication layer

So the production-safe assumption is:

- VPN-restricted deployment
- or reverse-proxy auth in front of the app

Do not treat the current API as internet-ready without adding authentication and authorization.

## Logging and Operations

At minimum, production operations should watch:

- service startup failures
- `/health` readiness failures
- Ollama connectivity failures
- retrieval failures
- generation failures
- indexing/deployment artifact mismatches

Useful signals to capture:

- backend startup time
- health-check latency
- answer latency
- search latency
- number of retrieved chunks
- Ollama errors

The current code already returns timing data from the answer flow, so the backend is a good place to add request logging later.

## Failure Modes to Plan For

The current deployment can fail in a few predictable ways:

- Ollama is not running
- the Ollama model is missing
- the vector-store manifest is missing
- the graph artifact is missing
- the embedding model cannot be loaded
- the frontend is served from a different origin and browser requests fail due to missing CORS
- a deployment ships code without matching artifacts

The good news is that `/health` surfaces several of these immediately because it warms the runtime assets on startup.

## Deployment Checklist

Before first rollout:

- choose a Python version and keep it pinned
- provision a Linux host with enough CPU or GPU for the chosen Ollama model
- install Docker Engine and Docker Compose
- copy the repo to the server
- configure `.env` with `CORTEXRAG_BIND_IP`
- copy or build `data/` and `storage/` artifacts on the server
- build and start the Compose stack
- pull the pinned Ollama model inside the `ollama` container
- verify `GET /health`
- verify one real `/search` request
- verify one real `/answer` request
- verify the browser UI loads and can submit a query

## Rollback Strategy

Define rollback around three things:

- code revision
- artifact set
- Ollama model tag

A clean rollback puts all three back into a compatible combination. Do not roll back code and leave newer graph or vector-store artifacts in place unless you know the schema is still compatible.

## Recommended Next Repo Improvements

The current deployment path is workable, but these repo changes would materially improve production readiness:

1. make data and storage paths environment-driven instead of repo-relative only
2. add backend CORS support only if a cross-origin deployment is actually needed
3. add a real auth layer or put one in the reverse proxy
4. add a `.env.example`
5. add optional reverse-proxy auth examples for VPN-less environments
6. add structured request logging in the API layer

## Non-Goals for the First Production Version

Avoid adding these until there is a concrete operational reason:

- multiple inference backends
- distributed vector databases
- autoscaling
- public ingestion APIs
- background orchestration systems
- multi-node graph infrastructure

The first production version should be simple, same-origin, and easy to debug.

## Summary

The clean production deployment for the current repo is:

- deploy a checked-out repository, not a wheel-only package
- keep `data/` and `storage/` in the repo layout the code expects
- build and run with Docker Compose
- run Ollama inside the private Compose network
- serve the built frontend from the `web` container on the same origin
- proxy `/health`, `/search`, `/answer`, and `/graph/*` to the backend
- bind the published web port only to localhost or a VPN interface

That architecture matches the code that exists now and avoids inventing deployment assumptions the repo does not yet support.
