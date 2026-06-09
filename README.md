# Document Intelligence RAG Platform

An end-to-end Retrieval-Augmented Generation platform for enterprise document
search and Q&A. It demonstrates a production-shaped data/ML stack:

- **PySpark** — parallel parse / chunk / embed of large document corpora
- **Snowflake** — curated chunk + query metadata (local SQLite fallback)
- **XGBoost** — document/query router that classifies into a category so
  retrieval can be filtered intelligently
- **FAISS** — vector store for top-k similarity search (NumPy fallback)
- **LLM** — grounded answer generation (stub / Groq / OpenAI / Bedrock providers)
- **FastAPI** — REST API for retrieval + answers, with `/metrics` for Prometheus
- **Docker + Kubernetes (EKS) + AWS** — containerization and deployment
- **MLflow + Prometheus** — experiment tracking and runtime monitoring
- **Power BI / Tableau** — BI dashboards over the metadata layer

> The project runs fully offline with safe fallbacks (hashing embedder, stub
> LLM, SQLite metadata) so you can demo it without any cloud credentials, then
> flip environment variables to use the real services.

## Architecture

```
            OFFLINE INGESTION
  documents ─► PySpark embed ─► vector store (+ Snowflake metadata)
                   │
              XGBoost router (classify + route)

            ONLINE QUERY
  user query ─► FastAPI retrieval ─► LLM answer ─► BI dashboard
                   ▲ retrieve context
                   └────────── vector store
```

## Quickstart (local, no credentials)

```bash
# 1. install
pip install -r requirements.txt        # or: make install

# 2. create sample corpus + labels
python scripts/seed_data.py            # or: make seed

# 3. train the XGBoost router
python scripts/train_router.py         # or: make train

# 4. ingest docs into the vector store
python scripts/ingest.py --docs data/sample_docs   # or: make ingest

# 5. run the API
uvicorn src.api.main:app --reload      # or: make run
```

Then open http://localhost:8000/docs and try:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many vacation days do I get?"}'
```

## Endpoints

| Method | Path          | Description                                  |
|--------|---------------|----------------------------------------------|
| GET    | `/health`     | Status, indexed chunk count, backends        |
| POST   | `/query`      | Route + retrieve + answer with sources       |
| GET    | `/categories` | Chunk counts per routed category             |
| GET    | `/metrics`    | Prometheus metrics                           |

## Tests

```bash
pytest -q          # or: make test
```

## Docker

```bash
docker compose up --build      # API on :8000, Prometheus on :9090
```

The image seeds, trains, and ingests the sample corpus at build time, so the
container is queryable immediately.

## Kubernetes (EKS)

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.example.yaml   # replace with a real secret
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

## Cluster-scale ingestion (PySpark)

```bash
spark-submit src/ingestion/spark_embed.py \
  --input s3://my-bucket/raw-docs \
  --output s3://my-bucket/chunks.parquet
```

## Configuration

Copy `.env.example` to `.env`. Key switches:

- `LLM_PROVIDER`: `stub` (default) | `groq` | `openai` | `bedrock`
  - For Groq: set `GROQ_API_KEY` and optionally `GROQ_MODEL` (default `llama-3.3-70b-versatile`). Groq gives very low-latency inference for open models via an OpenAI-compatible API.
- `EMBEDDING_BACKEND`: `hashing` (default) | `sentence-transformers`
- Snowflake vars: when set, metadata writes go to Snowflake instead of SQLite

## Project layout

```
config/            settings (pydantic)
src/api/           FastAPI app, routes, schemas
src/ingestion/     chunker, loaders, PySpark embedding job
src/embeddings/    embedder (sentence-transformers / hashing)
src/vectorstore/   FAISS / NumPy vector store
src/router/        XGBoost router (train + classify)
src/rag/           retriever, LLM client, pipeline
src/metadata/      Snowflake / SQLite metadata store
src/monitoring/    Prometheus metrics
scripts/           seed_data, ingest, train_router
tests/             unit + end-to-end tests
k8s/               Kubernetes manifests
dashboards/        Power BI / Tableau notes + SQL views
.github/workflows/ CI (tests + docker build)
```

## License

MIT
