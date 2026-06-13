# Document Intelligence RAG Platform

An end-to-end Retrieval-Augmented Generation (RAG) platform for document search
and Q&A. Ask a natural-language question and get an answer grounded in your
documents, with citations to the source passages, an XGBoost-routed category,
and the retrieval scores that produced it.

It pairs a Python ML/serving backend with a React frontend, and is built so the
whole thing runs locally with safe fallbacks — no cloud credentials required —
while keeping the production integrations implemented and pluggable.

## Tech stack

**Running in this demo (no credentials needed):**

- **FastAPI** — REST API for the RAG pipeline (`/query`, `/health`, `/categories`, `/metrics`)
- **FAISS** — vector store for top-k similarity search (NumPy fallback if FAISS is absent)
- **XGBoost** — query/document router that classifies into a category (TF-IDF → SVD → XGBoost)
- **Groq** — low-latency LLM inference (Llama 3.3) for grounded answer generation
- **Hashing embedder** — dependency-free embeddings so retrieval works offline
- **SQLite** — local metadata store (chunk + query logs)
- **Docker / Docker Compose** — containerized API + Prometheus
- **React + Vite** — frontend UI that calls the API and visualizes the answer, routed category, confidence, latency, and ranked sources
- **Prometheus** — runtime metrics
- **pytest** — unit + end-to-end tests

**Implemented and pluggable (built into the codebase; this demo runs the local fallbacks instead):**

- **PySpark** — parallel parse / chunk / embed job for cluster-scale ingestion (`src/ingestion/spark_embed.py`)
- **Snowflake** — production metadata store; activated by setting Snowflake credentials (falls back to SQLite here)
- **AWS Bedrock** — alternative LLM provider; selectable via `LLM_PROVIDER=bedrock`
- **MLflow** — experiment tracking for router training runs
- **Kubernetes (EKS)** — deployment manifests in `k8s/`
- **sentence-transformers** — higher-quality embeddings; selectable via `EMBEDDING_BACKEND`
- **Power BI / Tableau** — BI dashboard notes + SQL views over the metadata layer (`dashboards/`)

> These integrations are written and wired into the pipeline behind a fallback
> design (graceful degradation). To keep local setup simple and dependency-light,
> the demo runs FAISS + SQLite + the hashing embedder + Groq. Flip the relevant
> environment variables (and install the optional packages) to run the
> production paths.

## Architecture

```
            OFFLINE INGESTION
  documents ─► chunk + embed ─► vector store (+ metadata store)
                   │              (FAISS)     (SQLite / Snowflake)
              XGBoost router (classify category)

            ONLINE QUERY
  React UI ─► FastAPI ─► retrieve top-k ─► soft re-rank ─► LLM (Groq) ─► answer + sources
                              ▲ vector store
```

Retrieval runs on pure semantic similarity; the router's category is applied as
a soft re-ranking boost rather than a hard filter, so a misroute can never fully
exclude a relevant chunk.

## Backend — quickstart (local, no credentials)

```bash
pip install -r requirements.txt        # or: make install
python scripts/seed_data.py            # sample corpus + labels + router phrases
python scripts/train_router.py         # train the XGBoost router
python scripts/ingest.py --docs data/sample_docs   # build the vector store
uvicorn src.api.main:app --reload      # serve the API
```

Then open http://localhost:8000/docs, or query directly:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many vacation days do I get?"}'
```

## Frontend — React UI

A React + Vite interface for asking questions and viewing the answer with its
routed category, router confidence, response time, and ranked source documents.

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The backend must be
running, and CORS for the dev server is enabled in `src/api/main.py`. Requires
Node.js 18+.

## Docker

```bash
docker compose up --build      # API on :8000, Prometheus on :9090
```

The image seeds, trains, and ingests the sample corpus at build time, so the
container is queryable immediately. Provide secrets via an `.env` file
(`env_file: .env` in `docker-compose.yml`); never commit `.env`.

## Endpoints

| Method | Path          | Description                            |
|--------|---------------|----------------------------------------|
| GET    | `/health`     | Status, indexed chunk count, backends  |
| POST   | `/query`      | Route + retrieve + answer with sources |
| GET    | `/categories` | Chunk counts per routed category       |
| GET    | `/metrics`    | Prometheus metrics                     |

## Tests

```bash
pytest -q          # or: make test
```

## Configuration

Copy `.env.example` to `.env` (the real `.env` is gitignored — keep secrets out
of version control). Key switches:

- `LLM_PROVIDER`: `stub` (default) | `groq` | `openai` | `bedrock`
  - Groq: set `GROQ_API_KEY` and optionally `GROQ_MODEL` (default `llama-3.3-70b-versatile`).
- `EMBEDDING_BACKEND`: `hashing` (default) | `sentence-transformers`
- Snowflake vars: when set, metadata writes go to Snowflake instead of SQLite.

## Production integrations (optional)

These are implemented but not exercised in the local demo:

```bash
# Cluster-scale ingestion with PySpark
spark-submit src/ingestion/spark_embed.py \
  --input s3://my-bucket/raw-docs --output s3://my-bucket/chunks.parquet

# Deploy on Kubernetes (EKS)
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.example.yaml   # replace with a real secret
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

The heavy dependencies (pyspark, snowflake-connector-python, boto3, mlflow) are
commented out in `requirements.txt` so local install stays light; uncomment them
(ideally on Python 3.11/3.12) to run these paths.

## Project layout

```
config/            settings (pydantic)
src/api/           FastAPI app, routes, schemas (CORS enabled for the UI)
src/ingestion/     chunker, loaders, PySpark embedding job
src/embeddings/    embedder (hashing / sentence-transformers)
src/vectorstore/   FAISS / NumPy vector store
src/router/        XGBoost router (train + classify, native .ubj model format)
src/rag/           retriever, LLM client, pipeline
src/metadata/      SQLite / Snowflake metadata store
src/monitoring/    Prometheus metrics
scripts/           seed_data, ingest, train_router
tests/             unit + end-to-end tests
frontend/          React + Vite UI (calls the FastAPI /query endpoint)
k8s/               Kubernetes manifests
dashboards/        Power BI / Tableau notes + SQL views
.github/workflows/ CI (tests + docker build)
```

## License

MIT
