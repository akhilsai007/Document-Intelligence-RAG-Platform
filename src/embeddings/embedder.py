"""Text embedder with two backends:

- ``sentence-transformers``: real semantic embeddings (downloads a model).
- ``hashing``: a deterministic, dependency-free fallback (feature hashing +
  L2 normalization). Good enough to make the whole pipeline run and tests
  pass offline; swap to sentence-transformers in production.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import List

import numpy as np

from config.settings import settings

_TOKEN = re.compile(r"[a-z0-9]+")

# Small English stopword set: dropping these from the hashing embedder keeps
# content words dominant so short queries retrieve sensibly.
_STOPWORDS = frozenset(
    """a an the of to in on at for and or but is are was were be been being do
    does did how what when where which who whom why with without within into
    from by as it its this that these those i you he she they we my your our
    me us them his her their do don't can could should would will shall may
    might must have has had get got need want about over under more most some
    any all no not so than then there here""".split()
)


def _tokenize(text: str, drop_stopwords: bool = False) -> List[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        filtered = [t for t in tokens if t not in _STOPWORDS]
        # don't return empty if the whole string was stopwords
        return filtered or tokens
    return tokens


class Embedder:
    def __init__(self, backend: str | None = None, dim: int | None = None):
        self.backend = backend or settings.embedding_backend
        self.dim = dim or settings.embedding_dim
        self._st_model = None
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(settings.embedding_model)
                self.dim = self._st_model.get_sentence_embedding_dimension()
            except Exception:
                # graceful fallback if the lib/model is unavailable
                self.backend = "hashing"

    def _hash_embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text, drop_stopwords=True):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self._st_model is not None:
            emb = self._st_model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        return np.vstack([self._hash_embed_one(t) for t in texts]).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
