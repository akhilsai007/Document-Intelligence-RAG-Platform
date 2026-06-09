"""Prometheus metrics exposed at /metrics."""
from __future__ import annotations

from prometheus_client import Counter, Histogram, make_asgi_app

QUERY_COUNT = Counter(
    "rag_queries_total", "Total RAG queries", ["category"]
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds", "End-to-end query latency in seconds"
)

metrics_app = make_asgi_app()
