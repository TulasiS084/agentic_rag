"""
Hybrid Retrieval Engine.
Combines ChromaDB Dense Vector Search and BM25 Sparse Lexical Search using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from rag.config import CACHE_DIR, RRF_K, DEFAULT_DENSE_TOP_K, DEFAULT_SPARSE_TOP_K
from rag import embeddings, vectorstore

logger = logging.getLogger(__name__)

# In-memory cache for BM25 indexes: document_id -> {"bm25": BM25Okapi, "chunks": List[Dict]}
_bm25_indices: Dict[str, Dict[str, Any]] = {}


def _tokenize(text: str) -> List[str]:
    """Lightweight whitespace + alphanumeric tokenizer for BM25."""
    import re
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1]


def _get_bm25_cache_path(document_id: str) -> Path:
    return CACHE_DIR / f"{document_id}_bm25.pkl"


def build_bm25_index(document_id: str, chunks: List[Dict[str, Any]]) -> None:
    """Build BM25 index from chunks and cache to disk."""
    if not chunks:
        return

    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    data = {
        "bm25": bm25,
        "chunks": chunks,
    }
    _bm25_indices[document_id] = data

    cache_file = _get_bm25_cache_path(document_id)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Persisted BM25 index for '{document_id}' ({len(chunks)} chunks).")
    except Exception as e:
        logger.warning(f"Could not write BM25 cache for '{document_id}': {e}")


def load_bm25_index(document_id: str) -> bool:
    """Load BM25 index from memory or disk cache."""
    if document_id in _bm25_indices:
        return True

    cache_file = _get_bm25_cache_path(document_id)
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                _bm25_indices[document_id] = pickle.load(f)
            return True
        except Exception as e:
            logger.warning(f"Failed to read BM25 cache for {document_id}: {e}")

    # Fallback: rebuild from ChromaDB if chunks exist
    chunks = vectorstore.get_all_chunks(document_id)
    if chunks:
        build_bm25_index(document_id, chunks)
        return True

    return False


def remove_bm25_index(document_id: str) -> None:
    """Clear BM25 index from memory and disk cache."""
    _bm25_indices.pop(document_id, None)
    cache_file = _get_bm25_cache_path(document_id)
    if cache_file.exists():
        try:
            cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to unlink BM25 cache {cache_file}: {e}")


def sparse_search(
    document_ids: List[str],
    query: str,
    top_k: int = DEFAULT_SPARSE_TOP_K,
) -> List[Dict[str, Any]]:
    """Execute BM25 sparse search across active documents."""
    tokens = _tokenize(query)
    if not tokens:
        return []

    scored_chunks: List[tuple[Dict[str, Any], float]] = []

    for doc_id in document_ids:
        if not load_bm25_index(doc_id):
            continue

        index_entry = _bm25_indices[doc_id]
        bm25: BM25Okapi = index_entry["bm25"]
        chunks: List[Dict[str, Any]] = index_entry["chunks"]

        scores = bm25.get_scores(tokens)
        for chunk, score in zip(chunks, scores):
            if score > 0.0:
                scored_chunks.append((chunk, float(score)))

    # Sort descending by BM25 score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored_chunks[:top_k]:
        results.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}),
            "score": score,
            "retrieval_source": "sparse",
        })

    return results


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists (Dense + Sparse) using Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum( 1 / (k + rank_i(d)) )
    """
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            cid = item["chunk_id"]
            if not cid:
                continue

            rrf_val = 1.0 / (k + rank + 1)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_val

            if cid not in chunk_map:
                chunk_map[cid] = item.copy()

            if item.get("retrieval_source") == "dense":
                dense_scores[cid] = item.get("score", 0.0)
            elif item.get("retrieval_source") == "sparse":
                sparse_scores[cid] = item.get("score", 0.0)

    # Sort keys by fused score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    fused_results = []
    for cid in sorted_ids:
        entry = chunk_map[cid]
        entry["fused_score"] = round(fused_scores[cid], 5)
        entry["dense_score"] = round(dense_scores.get(cid, 0.0), 4)
        entry["sparse_score"] = round(sparse_scores.get(cid, 0.0), 4)
        entry["score"] = entry["fused_score"]
        fused_results.append(entry)

    return fused_results


def hybrid_search(
    query: str,
    document_ids: List[str],
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Execute Hybrid Retrieval (Dense Vector + Sparse BM25) and fuse candidates.

    Returns:
        {
            "results": List[Dict[str, Any]],
            "stats": {
                "dense_count": int,
                "sparse_count": int,
                "fused_count": int,
                "final_count": int,
            }
        }
    """
    if not document_ids or not query.strip():
        return {
            "results": [],
            "stats": {"dense_count": 0, "sparse_count": 0, "fused_count": 0, "final_count": 0},
        }

    # 1. Dense retrieval
    query_vector = embeddings.embed_query(query)
    dense_raw = vectorstore.query_multiple_collections(
        document_ids=document_ids,
        query_embedding=query_vector,
        n_results=dense_top_k,
    )
    for d in dense_raw:
        d["retrieval_source"] = "dense"

    # 2. Sparse retrieval
    sparse_raw = sparse_search(
        document_ids=document_ids,
        query=query,
        top_k=sparse_top_k,
    )

    # 3. Fuse candidates using RRF
    fused = reciprocal_rank_fusion([dense_raw, sparse_raw])
    selected = fused[:top_k]

    return {
        "results": selected,
        "stats": {
            "dense_count": len(dense_raw),
            "sparse_count": len(sparse_raw),
            "fused_count": len(fused),
            "final_count": len(selected),
        },
    }
