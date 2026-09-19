"""
ChromaDB Persistent Vector Store Manager.
Handles collections, upserts, vector queries, multi-collection aggregation, and document deletion.
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb

from rag.config import CHROMA_DB_DIR, CHROMA_COLLECTION_PREFIX

logger = logging.getLogger(__name__)

# Singleton persistent Chroma client
_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Retrieve or initialize the persistent ChromaDB client."""
    global _client
    if _client is None:
        logger.info(f"Connecting to persistent ChromaDB at {CHROMA_DB_DIR}")
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _client


def collection_name(document_id: str) -> str:
    """Deterministic collection name meeting Chroma naming constraints."""
    # Chroma requires 3-63 characters, alphanumeric or underscores/hyphens
    safe = document_id.replace("-", "_").replace(".", "_")[:50]
    return f"{CHROMA_COLLECTION_PREFIX}_{safe}"


def compute_file_hash(file_path: str | Path) -> str:
    """Generate SHA256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def add_chunks(
    document_id: str,
    chunk_ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    """
    Store chunk texts, embeddings, and sanitized metadatas in ChromaDB.
    """
    if not chunk_ids:
        return

    client = get_chroma_client()
    col_name = collection_name(document_id)

    # Sanitize metadata: ChromaDB only permits str, int, float, bool
    sanitized_metas = []
    for meta in metadatas:
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        sanitized_metas.append(clean)

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in safe batches to avoid hitting any payload limits
    batch_size = 500
    total = len(chunk_ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.upsert(
            ids=chunk_ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=sanitized_metas[i:end],
        )

    logger.info(f"Upserted {len(chunk_ids)} chunks into ChromaDB collection '{col_name}'")


def query_collection(
    document_id: str,
    query_embedding: List[float],
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    """Dense retrieval for a single document collection."""
    client = get_chroma_client()
    col_name = collection_name(document_id)

    try:
        collection = client.get_collection(name=col_name)
    except Exception:
        logger.warning(f"Collection not found: {col_name}")
        return []

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    if results and results["ids"] and results["ids"][0]:
        for idx, cid in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx] if results.get("distances") else 0.0
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, 1.0 - distance)
            hits.append({
                "chunk_id": cid,
                "text": results["documents"][0][idx] if results.get("documents") else "",
                "metadata": results["metadatas"][0][idx] if results.get("metadatas") else {},
                "score": float(similarity),
            })

    return hits


def query_multiple_collections(
    document_ids: List[str],
    query_embedding: List[float],
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    """Dense retrieval across multiple document collections."""
    all_hits: List[Dict[str, Any]] = []
    for doc_id in document_ids:
        hits = query_collection(doc_id, query_embedding, n_results)
        all_hits.extend(hits)

    # Sort descending by similarity score
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:n_results]


def delete_collection(document_id: str) -> None:
    """Safely delete a document's vector collection."""
    client = get_chroma_client()
    col_name = collection_name(document_id)
    try:
        client.delete_collection(name=col_name)
        logger.info(f"Deleted collection: {col_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection {col_name}: {e}")


def collection_exists(document_id: str) -> bool:
    """Check if a document collection exists and has vectors."""
    client = get_chroma_client()
    col_name = collection_name(document_id)
    try:
        col = client.get_collection(name=col_name)
        return col.count() > 0
    except Exception:
        return False


def get_all_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all stored chunks for a document (used to rebuild BM25)."""
    client = get_chroma_client()
    col_name = collection_name(document_id)
    try:
        collection = client.get_collection(name=col_name)
        count = collection.count()
        if count == 0:
            return []
        results = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )
        chunks = []
        for idx, cid in enumerate(results["ids"]):
            chunks.append({
                "chunk_id": cid,
                "text": results["documents"][idx] if results.get("documents") else "",
                "metadata": results["metadatas"][idx] if results.get("metadatas") else {},
            })
        return chunks
    except Exception:
        return []


def get_vector_store_stats() -> Dict[str, Any]:
    """Return summary statistics of all collections in ChromaDB."""
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        total_vectors = sum(c.count() for c in collections)
        return {
            "total_collections": len(collections),
            "total_vectors": total_vectors,
            "collections": [c.name for c in collections],
        }
    except Exception as e:
        logger.error(f"Error fetching ChromaDB stats: {e}")
        return {"total_collections": 0, "total_vectors": 0, "collections": []}
