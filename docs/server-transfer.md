# Docker + VPN Transfer Runbook

This is the preferred first deployment for CortexRAG: one server, Docker Compose, Ollama in the private Compose network, and the web port reachable only through your VPN or localhost.

This is safe enough for a private first rollout if the VPN is the access boundary. The app still has no built-in user authentication, so do not publish the Compose web port on a public interface.

## Architecture

Compose runs three services:

- `web`: Nginx serving the built React frontend and proxying API routes
- `api`: FastAPI/Uvicorn backend
- `ollama`: local model runtime, not exposed on the host

Only `web` publishes a host port. By default it binds to `127.0.0.1:8080`; for VPN access, bind it to your VPN interface IP.

## 1. Prepare The Server

Install Docker Engine and the Docker Compose plugin on the server. On Ubuntu/Debian, use Docker's official install path, then confirm:

```bash
docker --version
docker compose version
```

Create the app directory:

```bash
sudo mkdir -p /opt/cortexrag/app
sudo chown -R "$USER":"$USER" /opt/cortexrag
```

## 2. Transfer The Project

Either clone the repository on the server:

```bash
cd /opt/cortexrag
git clone YOUR_REPO_URL app
cd app
```

or copy the local working tree from your local repository root:

```bash
rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude frontend/node_modules \
  --exclude frontend/dist \
  ./ USER@SERVER:/opt/cortexrag/app/
```

Replace `USER@SERVER` with your SSH user and server host.

Generated corpus and vector-store files are ignored by Git, but `rsync` will copy your local `data/` and `storage/` files unless you exclude them. This local checkout currently has only placeholder files there, so you will need real artifacts or raw Confluence exports before the app can answer queries.

If you cloned the repo and need to copy existing local artifacts, run this from your local repository root:

```bash
rsync -az data/ USER@SERVER:/opt/cortexrag/app/data/
rsync -az storage/ USER@SERVER:/opt/cortexrag/app/storage/
```

If you only want to copy raw Confluence exports and build everything on the server:

```bash
rsync -az data/raw/confluence/ USER@SERVER:/opt/cortexrag/app/data/raw/confluence/
```

On Windows PowerShell without `rsync`, use `scp`:

```powershell
scp -r .\data\ USER@SERVER:/opt/cortexrag/app/
scp -r .\storage\ USER@SERVER:/opt/cortexrag/app/
```

## 3. Configure VPN Binding

On the server, create `/opt/cortexrag/app/.env` for Compose:

```env
CORTEXRAG_BIND_IP=127.0.0.1
```

For VPN access, replace `127.0.0.1` with the server's VPN IP, for example a Tailscale `100.x.y.z` address:

```env
CORTEXRAG_BIND_IP=100.x.y.z
```

Keep the firewall closed on public interfaces. The app should be reachable only through VPN as:

```text
http://VPN_IP:8080
```

## 4. Build Images And Pull The Model

On the server:

```bash
cd /opt/cortexrag/app
docker compose build
docker compose up -d ollama
```

Pull the default Ollama model into the `ollama` volume:

```bash
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama list
```

## 5. Build Or Verify Artifacts

If artifacts were copied, verify:

```bash
ls /opt/cortexrag/app/storage/chroma
```

If you need to build artifacts on the server, copy raw Confluence exports to `data/raw/confluence/`, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.index.yml run --rm api python scripts/preprocess_confluence_exports.py
docker compose -f docker-compose.yml -f docker-compose.index.yml run --rm api python scripts/chunk_confluence_exports.py
docker compose -f docker-compose.yml -f docker-compose.index.yml run --rm api python scripts/embed_confluence_chunks.py
docker compose -f docker-compose.yml -f docker-compose.index.yml run --rm api python -m cortex_rag build-vector-store --with-graph
docker compose restart api
```

The normal Compose file mounts `data/` and `storage/` read-only for the running API. The `docker-compose.index.yml` override makes those mounts writable only for these controlled indexing runs.

## 6. Start The App

```bash
docker compose up -d
```

## 7. Final Checks

From the server:

```bash
curl -f http://127.0.0.1:8080/health
docker compose ps
docker compose logs --tail=100 api
```

From a VPN-connected client:

```bash
curl -f http://VPN_IP:8080/health
```

Then open:

```text
http://VPN_IP:8080
```

## Operations

Useful commands:

```bash
docker compose logs -f api
docker compose logs -f web
docker compose restart api
docker compose pull
docker compose up -d --build
```

Rollback should keep these in sync:

- app code
- `data/` and `storage/` artifacts
- Ollama model tag

## Security Notes

- Docker is packaging, not authentication. The VPN is the security boundary.
- Do not expose `8080` on `0.0.0.0` unless a firewall restricts it to VPN clients.
- Ollama is internal to Compose and should stay unexposed.
- Same-origin proxying is preserved, so the frontend does not need backend CORS.
