"""
Document Chunking and Statistics Computation.
Uses LangChain's RecursiveCharacterTextSplitter while maintaining full metadata traceability.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_id: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split extracted document pages into semantic chunks.

    Returns:
        (chunks, stats)
        where chunks is a list of chunk dicts and stats is a summary dictionary.
    """
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
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "page_number": page_num,
                "document_id": document_id,
                "char_length": len(split_text),
            }

            chunks.append({
                "chunk_id": chunk_id,
                "text": split_text,
                "chunk_index": chunk_idx,
                "page_number": page_num,
                "document_id": document_id,
                "metadata": chunk_metadata,
            })
            chunk_idx += 1

    # Ingestion Statistics
    total_chunks = len(chunks)
    avg_chunk_size = round(total_characters / total_chunks, 1) if total_chunks > 0 else 0
    estimated_tokens = int(total_words * 1.3)  # Standard rule of thumb: ~1.3 tokens per word

    stats = {
        "total_pages": len(pages),
        "total_characters": total_characters,
        "total_words": total_words,
        "estimated_tokens": estimated_tokens,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    logger.info(
        f"Chunked {len(pages)} pages into {total_chunks} chunks. "
        f"Avg size: {avg_chunk_size} chars. Est tokens: {estimated_tokens}"
    )

    return chunks, stats
