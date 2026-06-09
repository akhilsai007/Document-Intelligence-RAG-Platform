"""Simple, deterministic text chunker with overlap (word-based)."""
from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
