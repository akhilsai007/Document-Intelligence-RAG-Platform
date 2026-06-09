"""Load raw documents from a directory. Supports .txt/.md natively and .pdf
when pypdf is installed. Each document carries lightweight metadata."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def load_documents(directory: str) -> List[Dict]:
    docs: List[Dict] = []
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if ext in {".txt", ".md"}:
                text = _read_text_file(path)
            elif ext == ".pdf":
                text = _read_pdf(path)
            else:
                continue
            if not text.strip():
                continue
            docs.append(
                {
                    "doc_id": os.path.splitext(name)[0],
                    "source": path,
                    "text": text,
                    "ext": ext,
                }
            )
    return docs


def iter_documents(directory: str) -> Iterator[Dict]:
    yield from load_documents(directory)
