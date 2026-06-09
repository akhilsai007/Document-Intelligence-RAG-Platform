"""Vector store backed by FAISS (cosine via inner product on L2-normalized
vectors). Falls back to a NumPy brute-force index if FAISS is unavailable.
Persists to disk so ingestion and serving are decoupled."""
from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from config.settings import settings
from src.embeddings import get_embedder
from src.vectorstore.base import Chunk

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self.chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.chunks.extend(chunks)
        if _HAS_FAISS:
            self._index.add(vectors)
        else:
            self._vectors = (
                vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            )

    def search(
        self, query_vec: np.ndarray, k: int = 4, category: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1).astype(np.float32))
        # over-fetch so category filtering still returns k results
        fetch = min(len(self.chunks), max(k * 5, k))
        if _HAS_FAISS:
            scores, idx = self._index.search(q, fetch)
            pairs = list(zip(idx[0], scores[0]))
        else:
            sims = (self._vectors @ q[0])
            order = np.argsort(-sims)[:fetch]
            pairs = [(int(i), float(sims[i])) for i in order]
        results: List[Tuple[Chunk, float]] = []
        for i, score in pairs:
            if i < 0:
                continue
            chunk = self.chunks[i]
            if category and chunk.category != category:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results

    def save(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {"dim": self.dim, "has_faiss": _HAS_FAISS, "count": len(self.chunks)}
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f)
        if _HAS_FAISS:
            faiss.write_index(self._index, os.path.join(self.path, "index.faiss"))
        elif self._vectors is not None:
            np.save(os.path.join(self.path, "vectors.npy"), self._vectors)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim=dim, path=path)
        chunks_p = os.path.join(path, "chunks.pkl")
        if not os.path.exists(chunks_p):
            return store
        with open(chunks_p, "rb") as f:
            store.chunks = pickle.load(f)
        if _HAS_FAISS and os.path.exists(os.path.join(path, "index.faiss")):
            store._index = faiss.read_index(os.path.join(path, "index.faiss"))
        elif os.path.exists(os.path.join(path, "vectors.npy")):
            store._vectors = np.load(os.path.join(path, "vectors.npy"))
        return store


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    dim = get_embedder().dim
    return VectorStore.load(settings.vector_store_path, dim=dim)
