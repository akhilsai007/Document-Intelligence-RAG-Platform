"""FastAPI application entrypoint for the document intelligence RAG platform."""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.monitoring import metrics_app

app = FastAPI(
    title="Document Intelligence RAG Platform",
    version="0.1.0",
    description="Spark ingestion + Snowflake metadata + XGBoost router + "
    "FastAPI retrieval + LLM answers.",
)

app.include_router(router)
app.mount("/metrics", metrics_app)


@app.get("/", tags=["ops"])
def root():
    return {"service": "rag-doc-intelligence", "docs": "/docs", "health": "/health"}
