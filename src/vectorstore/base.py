"""Vector store record schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
