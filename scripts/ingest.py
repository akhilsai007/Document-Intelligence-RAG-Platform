"""Local (non-Spark) ingestion: load docs -> route category -> chunk ->
embed -> build vector store -> write metadata. Use this for dev and demos;
use src/ingestion/spark_embed.py for cluster-scale ingestion."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.embeddings import get_embedder
from src.ingestion import chunk_text, load_documents
from src.metadata import get_metadata_store
from src.router import get_router
from src.vectorstore.base import Chunk
from src.vectorstore.faiss_store import VectorStore


def main(docs_dir: str, labels_path: str | None):
    labels = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)

    embedder = get_embedder()
    router = get_router()
    metadata = get_metadata_store()
    store = VectorStore(dim=embedder.dim, path=settings.vector_store_path)

    documents = load_documents(docs_dir)
    all_chunks, meta_rows = [], []
    for doc in documents:
        # prefer ground-truth label; otherwise let the router classify
        category = labels.get(doc["doc_id"])
        if category is None:
            category, _ = router.predict(doc["text"])
        for i, text in enumerate(chunk_text(doc["text"])):
            cid = f"{doc['doc_id']}::{i}"
            all_chunks.append(
                Chunk(chunk_id=cid, doc_id=doc["doc_id"], text=text,
                      category=category, metadata={"source": doc["source"]})
            )
            meta_rows.append(
                {"chunk_id": cid, "doc_id": doc["doc_id"], "category": category,
                 "source": doc["source"], "char_len": len(text)}
            )

    if not all_chunks:
        print("No documents found to ingest.")
        return

    vectors = embedder.encode([c.text for c in all_chunks])
    store.add(all_chunks, vectors)
    store.save()
    metadata.upsert_chunks(meta_rows)
    print(f"Ingested {len(documents)} docs -> {len(all_chunks)} chunks "
          f"into {settings.vector_store_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/sample_docs")
    p.add_argument("--labels", default="data/labels.json")
    args = p.parse_args()
    main(args.docs, args.labels)
