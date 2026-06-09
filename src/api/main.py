"""FastAPI application entrypoint for the document intelligence RAG platform."""
from __future__ import annotations

from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore

from src.api.routes import router
from src.monitoring import metrics_app

app = FastAPI(
    title="Document Intelligence RAG Platform",
    version="0.1.0",
    description="Spark ingestion + Snowflake metadata + XGBoost router + "
    "FastAPI retrieval + LLM answers.",
)

# Allow the React dev server to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/metrics", metrics_app)


@app.get("/", tags=["ops"])
def root():
    return {"service": "rag-doc-intelligence", "docs": "/docs", "health": "/health"}