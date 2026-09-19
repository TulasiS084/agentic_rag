"""
Document Chunking and Statistics Computation.
Uses LangChain's RecursiveCharacterTextSplitter while maintaining full metadata traceability.
Automatically enforces 10% chunk overlap: overlap = int(chunk_size * 0.10).
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)


def compute_chunk_overlap(chunk_size: int) -> int:
    """Calculate chunk overlap as exactly 10% of chunk size."""
    return max(1, int(chunk_size * 0.10))


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: Optional[int] = None,
    document_id: str = "",
    document_name: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split extracted document pages into semantic chunks.
    Preserves page number, document ID, and document name.
    Enforces automatic 10% overlap: overlap = int(chunk_size * 0.10).

    Returns:
        (chunks, stats)
    """
    if chunk_overlap is None:
        chunk_overlap = compute_chunk_overlap(chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    total_characters = 0
    total_words = 0
    chunk_idx = 0

    for page_data in pages:
        page_text = page_data.get("text", "")
        page_num = page_data.get("page", 1)
        page_meta = page_data.get("metadata", {})
        doc_name = document_name or page_meta.get("source", "Document")

        if not page_text.strip():
            continue

        total_characters += len(page_text)
        total_words += len(page_text.split())

        splits = splitter.split_text(page_text)

        for split_text in splits:
            if not split_text.strip():
                continue

            chunk_id = f"{document_id}_c{chunk_idx}"
            chunk_metadata = {
                **page_meta,
                "document_id": document_id,
                "document_name": doc_name,
                "source": doc_name,
                "page_number": page_num,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "char_length": len(split_text),
            }

            chunks.append({
                "chunk_id": chunk_id,
                "text": split_text,
                "chunk_index": chunk_idx,
                "page_number": page_num,
                "document_id": document_id,
                "document_name": doc_name,
                "metadata": chunk_metadata,
            })
            chunk_idx += 1

    total_chunks = len(chunks)
    avg_chunk_size = round(total_characters / total_chunks, 1) if total_chunks > 0 else 0
    estimated_tokens = int(total_words * 1.3)

    stats = {
        "total_pages": len(pages),
        "total_characters": total_characters,
        "total_words": total_words,
        "estimated_tokens": estimated_tokens,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "overlap_percentage": "10%",
    }

    logger.info(
        f"Chunked {len(pages)} pages into {total_chunks} chunks. "
        f"(size={chunk_size}, overlap={chunk_overlap} [10%])"
    )

    return chunks, stats
