"""Retriever: embed the query and pull the most similar chunks from the
vector store. Retrieval is by pure semantic similarity; the router's category
is applied as a soft re-ranking boost in the pipeline rather than a hard
filter, so an imperfect route can never exclude a highly relevant chunk."""
from __future__ import annotations

from typing import List, Tuple

from src.embeddings import get_embedder
from src.vectorstore import get_vector_store
from src.vectorstore.base import Chunk


class Retriever:
    def __init__(self):
        self.embedder = get_embedder()
        self.store = get_vector_store()

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        qvec = self.embedder.encode([query])[0]
        return self.store.search(qvec, k=k, category=None)
