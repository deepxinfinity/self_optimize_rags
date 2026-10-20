"""Overlapping word-window chunking. Sizes are in whitespace-separated words, not model tokens."""

from __future__ import annotations

from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    chunk_id: str    # <doc_id>::<ordinal>
    doc_id: str
    text: str
    ordinal: int


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({overlap}) must be < chunk_size ({chunk_size})")
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for i, text in enumerate(_chunk_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{i}", doc_id=doc.doc_id,
                                text=text, ordinal=i))
    return chunks
