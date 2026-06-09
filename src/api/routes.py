from __future__ import annotations

from fastapi import APIRouter, Depends

from config.settings import settings
from src.api.deps import pipeline_dep, store_dep
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.metadata import get_metadata_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(store=Depends(store_dep)):
    return HealthResponse(
        status="ok",
        indexed_chunks=len(store.chunks),
        embedding_backend=settings.embedding_backend,
        llm_provider=settings.llm_provider,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, pipeline=Depends(pipeline_dep)):
    result = pipeline.answer(req.question, k=req.top_k, use_router=req.use_router)
    return QueryResponse(**result.to_dict())


@router.get("/categories", tags=["rag"])
def categories():
    return get_metadata_store().category_counts()
