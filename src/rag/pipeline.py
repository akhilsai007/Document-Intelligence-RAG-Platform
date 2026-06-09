"""End-to-end RAG pipeline: route -> retrieve -> generate, with metadata
logging and Prometheus metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

from config.settings import settings
from src.metadata import get_metadata_store
from src.monitoring import QUERY_COUNT, QUERY_LATENCY
from src.rag.llm import LLMClient
from src.rag.retriever import Retriever
from src.router import get_router


@dataclass
class Source:
    doc_id: str
    chunk_id: str
    score: float
    category: str
    snippet: str


@dataclass
class RAGAnswer:
    answer: str
    category: str
    router_confidence: float
    sources: List[Source] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "category": self.category,
            "router_confidence": self.router_confidence,
            "latency_ms": self.latency_ms,
            "sources": [s.__dict__ for s in self.sources],
        }


class RAGPipeline:
    def __init__(self):
        self.router = get_router()
        self.retriever = Retriever()
        self.llm = LLMClient()
        self.metadata = get_metadata_store()

    def answer(self, question: str, k: int | None = None, use_router: bool = True) -> RAGAnswer:
        k = k or settings.top_k
        start = time.perf_counter()

        category, confidence = ("general", 0.0)
        if use_router:
            category, confidence = self.router.predict(question)

        # Over-fetch by pure similarity, then let the router gently boost
        # in-category chunks. This uses the routing signal without letting a
        # misroute exclude a highly relevant chunk (no hard filtering).
        candidates = self.retriever.retrieve(question, k=max(k * 3, 8))
        boost = 0.05 if (use_router and confidence >= 0.5) else 0.0
        ranked = sorted(
            candidates,
            key=lambda cs: cs[1] + (boost if cs[0].category == category else 0.0),
            reverse=True,
        )
        hits = ranked[:k]

        contexts = [c.text for c, _ in hits]
        sources = [c.doc_id for c, _ in hits]
        answer_text = self.llm.generate(question, contexts, sources)

        latency_ms = (time.perf_counter() - start) * 1000.0
        QUERY_COUNT.labels(category=category).inc()
        QUERY_LATENCY.observe(latency_ms / 1000.0)
        try:
            self.metadata.log_query(question, category, len(hits), latency_ms)
        except Exception:
            pass

        return RAGAnswer(
            answer=answer_text,
            category=category,
            router_confidence=confidence,
            latency_ms=round(latency_ms, 2),
            sources=[
                Source(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    score=round(score, 4),
                    category=c.category,
                    snippet=c.text[:200],
                )
                for c, score in hits
            ],
        )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()
