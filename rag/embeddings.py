"""
Embedding Generation using SentenceTransformers (all-MiniLM-L6-v2).
Singleton model loading + batch encoding for high efficiency and caching.
"""
from __future__ import annotations
import logging
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Singleton model instance
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load or retrieve the cached SentenceTransformer model."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on device: {device}")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info("SentenceTransformer model loaded successfully.")
    return _model


def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """
    Generate vector embeddings for a list of text strings.

    Returns:
        List of floating point embedding vectors.
    """
    if not texts:
        return []

    model = get_embedding_model()
    logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalized for cosine similarity
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Generate normalized embedding for a single user query."""
    model = get_embedding_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()
