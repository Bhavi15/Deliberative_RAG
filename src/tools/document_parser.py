"""Document upload and parsing tool.

Parses user-uploaded documents (PDF, text, markdown) into passages
that can be fed directly into the conflict detection pipeline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# Maximum characters per chunk when splitting large documents
_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 200


def parse_document(file_path: str) -> str:
    """Parse a document file and return its text content.

    Supports PDF, plain text, and markdown files.

    Args:
        file_path: Absolute path to the document file.

    Returns:
        Formatted string of extracted text passages with metadata.
    """
    path = Path(file_path)
    if not path.exists():
        return f"[File not found: {file_path}]"

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            text = _parse_pdf(path)
        elif ext in {".txt", ".md", ".markdown", ".rst"}:
            text = path.read_text(encoding="utf-8")
        elif ext in {".html", ".htm"}:
            text = _parse_html(path)
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as exc:
        log.error("document_parse_failed", path=str(path), error=str(exc))
        return f"[Failed to parse {path.name}: {exc}]"

    if not text.strip():
        return f"[Document is empty: {path.name}]"

    chunks = _chunk_text(text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
    log.info("document_parsed", file=path.name, chunks=len(chunks))

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i} from {path.name}]\n{chunk}")

    return "\n\n---\n\n".join(parts)


def parse_document_structured(file_path: str) -> list[dict[str, Any]]:
    """Parse a document into structured passage dicts for pipeline integration.

    Args:
        file_path: Absolute path to the document file.

    Returns:
        List of document dicts compatible with the retrieval pipeline.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            text = _parse_pdf(path)
        elif ext in {".txt", ".md", ".markdown", ".rst"}:
            text = path.read_text(encoding="utf-8")
        elif ext in {".html", ".htm"}:
            text = _parse_html(path)
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as exc:
        log.error("document_parse_structured_failed", path=str(path), error=str(exc))
        return []

    if not text.strip():
        return []

    chunks = _chunk_text(text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
    file_hash = hashlib.md5(path.name.encode()).hexdigest()[:8]

    documents: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"upload_{file_hash}_{i:03d}",
            "text": chunk,
            "score": 1.0,  # user-uploaded docs get max relevance
            "source_type": "uploaded_document",
            "publication_date": None,
            "source": f"upload:{path.name}",
            "title": f"{path.name} (passage {i + 1})",
        })

    log.info("document_parsed_structured", file=path.name, passages=len(documents))
    return documents


def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF file using pymupdf.

    Args:
        path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def _parse_html(path: Path) -> str:
    """Extract text from an HTML file by stripping tags.

    Args:
        path: Path to the HTML file.

    Returns:
        Plain text content.
    """
    import re
    raw = path.read_text(encoding="utf-8")
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Args:
        text: Full document text.
        chunk_size: Target characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Split on sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Keep overlap: walk back from end until we have ~overlap chars
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s)

            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]
