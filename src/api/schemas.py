from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the PTO policy?"])
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_router: bool = True


class SourceModel(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    category: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    category: str
    router_confidence: float
    latency_ms: float
    sources: List[SourceModel]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    embedding_backend: str
    llm_provider: str
