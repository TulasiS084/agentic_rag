"""
Cross-Encoder Reranker using SentenceTransformers.
Scores retrieved candidates against the query and returns top-N ranked chunks.
Includes a lightweight fallback ranking mechanism for high-efficiency CPU execution.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder

from rag.config import RERANKER_MODEL, DEFAULT_TOP_N_RERANK

logger = logging.getLogger(__name__)

# Singleton CrossEncoder instance
_reranker_model: Optional[CrossEncoder] = None
_reranker_failed: bool = False


def get_reranker_model() -> Optional[CrossEncoder]:
    """Retrieve or load cached CrossEncoder model."""
    global _reranker_model, _reranker_failed
    if _reranker_model is None and not _reranker_failed:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CrossEncoder: {RERANKER_MODEL} on {device}")
            _reranker_model = CrossEncoder(RERANKER_MODEL, device=device)
            logger.info("CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder model ({e}). Using lightweight fallback reranker.")
            _reranker_failed = True
    return _reranker_model


def fallback_rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N_RERANK,
) -> List[Dict[str, Any]]:
    """
    Lightweight fallback ranking mechanism based on normalized lexical overlap + RRF scores.
    Fast on CPU without requiring cross-encoder neural forward passes.
    """
    if not chunks:
        return []

    import re
    query_terms = set(re.findall(r"\w+", query.lower()))

    scored_chunks = []
    for idx, c in enumerate(chunks):
        item = c.copy()
        text_terms = set(re.findall(r"\w+", item["text"].lower()))
        overlap = len(query_terms.intersection(text_terms)) / max(1, len(query_terms))
        fused_score = item.get("fused_score", item.get("score", 0.0))
        # Combined lightweight score
        combined_score = (0.6 * fused_score) + (0.4 * overlap)
        item["rerank_score"] = round(combined_score, 4)
        scored_chunks.append(item)

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_results = scored_chunks[:top_n]
    for rank, chunk in enumerate(top_results, start=1):
        chunk["rank"] = rank

    return top_results


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N_RERANK,
    use_cross_encoder: bool = True,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks by relevance to query.

    Args:
        query: User question.
        chunks: List of retrieved chunks.
        top_n: Number of top results to return.
        use_cross_encoder: If False, uses fast lightweight fallback ranking.

    Returns:
        Top-N chunks with 'rerank_score' and 'rank' added.
    """
    if not chunks:
        return []

    if not use_cross_encoder:
        return fallback_rerank(query, chunks, top_n=top_n)

    model = get_reranker_model()
    if model is None:
        return fallback_rerank(query, chunks, top_n=top_n)

    try:
        # Create (query, chunk_text) pairs
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)

        scored_chunks = []
        for c, score in zip(chunks, scores):
            item = c.copy()
            item["rerank_score"] = round(float(score), 4)
            scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = scored_chunks[:top_n]

        for rank, chunk in enumerate(top_results, start=1):
            chunk["rank"] = rank

        logger.info(
            f"CrossEncoder reranked {len(chunks)} chunks -> top {len(top_results)}. "
            f"Top score: {top_results[0]['rerank_score'] if top_results else 'N/A'}"
        )
        return top_results
    except Exception as e:
        logger.error(f"CrossEncoder prediction error: {e}. Using fallback.")
        return fallback_rerank(query, chunks, top_n=top_n)
