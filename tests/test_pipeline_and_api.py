"""End-to-end: seed -> ingest -> query through the FastAPI app."""
import importlib

from fastapi.testclient import TestClient


def _seed_and_ingest(tmp_path):
    from scripts import seed_data, ingest

    docs = tmp_path / "docs"
    labels = tmp_path / "labels.json"
    seed_data.main(out_dir=str(docs), labels_path=str(labels))
    ingest.main(str(docs), str(labels))


def test_query_endpoint_returns_grounded_answer(tmp_path):
    _seed_and_ingest(tmp_path)

    # rebuild singletons now that the store on disk exists
    import src.vectorstore.faiss_store as vs
    import src.rag.pipeline as pl

    vs.get_vector_store.cache_clear()
    pl.get_pipeline.cache_clear()

    from src.api.main import app

    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["indexed_chunks"] > 0

    resp = client.post("/query", json={"question": "How many vacation days do I get?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1
    assert body["latency_ms"] >= 0
