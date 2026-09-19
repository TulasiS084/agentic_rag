"""
Verification test for the Agentic RAG pipeline.
Tests document loading, chunking, embeddings, ChromaDB, BM25, reranking, and agent routing.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from rag import loaders, chunker, embeddings, vectorstore, retriever, reranker, agent, voice

def run_tests():
    print("=" * 60)
    print("STARTING PIPELINE VERIFICATION TESTS")
    print("=" * 60)

    # 1. Create a sample test document
    sample_text = """
    Artificial Intelligence and Machine Learning Overview

    Chapter 1: Introduction to AI
    Artificial Intelligence (AI) is a branch of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation.

    Chapter 2: Machine Learning Foundations
    Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Supervised learning algorithms learn from labeled data, whereas unsupervised learning algorithms discover hidden patterns in unlabeled data. Reinforcement learning trains agents through feedback loops of rewards and penalties.

    Chapter 3: Deep Learning and Neural Networks
    Deep Learning utilizes artificial neural networks with multiple layers to model complex representations. Convolutional Neural Networks (CNNs) excel at computer vision tasks, Recurrent Neural Networks (RNNs) and Transformers revolutionize Natural Language Processing (NLP), and Generative Adversarial Networks (GANs) generate synthetic data.

    Chapter 4: Agentic RAG Architecture
    Retrieval-Augmented Generation (RAG) grounds language models on private knowledge bases. Agentic RAG introduces intelligent decision-making, such as query classification, multi-hop retrieval, Reciprocal Rank Fusion (RRF) between dense and sparse retrievers, and cross-encoder reranking to ensure high precision and recall.
    """

    test_file = Path(__file__).parent / "test_doc.txt"
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(sample_text.strip(), encoding="utf-8")
    print(f"[OK] Created test document: {test_file}")

    # 2. Test Document Extraction
    start = time.time()
    pages = loaders.load_document(test_file)
    load_time = time.time() - start
    print(f"[OK] Document loaded: {len(pages)} pages in {load_time:.3f}s")
    assert len(pages) > 0, "No pages extracted"

    # 3. Test Chunking
    chunks, stats = chunker.chunk_pages(pages, chunk_size=300, chunk_overlap=50, document_id="test_doc")
    print(f"[OK] Chunked into {len(chunks)} chunks. Stats: {stats}")
    assert len(chunks) >= 3, "Expected at least 3 chunks"

    # 4. Test Embeddings
    sample_texts = [c["text"] for c in chunks]
    emb_vectors = embeddings.embed_texts(sample_texts)
    print(f"[OK] Generated {len(emb_vectors)} embeddings. Vector dim: {len(emb_vectors[0])}")
    assert len(emb_vectors) == len(chunks), "Embedding count mismatch"

    # 5. Test ChromaDB Upsert & Query
    chunk_ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vectorstore.add_chunks(
        document_id="test_doc",
        chunk_ids=chunk_ids,
        texts=sample_texts,
        embeddings=emb_vectors,
        metadatas=metadatas,
    )
    assert vectorstore.collection_exists("test_doc"), "Collection should exist in ChromaDB"
    print("[OK] ChromaDB upsert successful")

    q_vec = embeddings.embed_query("What are the neural network architectures in Deep Learning?")
    dense_hits = vectorstore.query_collection("test_doc", q_vec, n_results=3)
    print(f"[OK] Dense query returned {len(dense_hits)} hits. Top score: {dense_hits[0]['score']:.4f}")

    # 6. Test BM25 Sparse Index & Search
    retriever.build_bm25_index("test_doc", chunks)
    sparse_hits = retriever.sparse_search(["test_doc"], "neural networks Transformers CNN", top_k=3)
    print(f"[OK] BM25 search returned {len(sparse_hits)} hits. Top score: {sparse_hits[0]['score']:.4f}")

    # 7. Test Hybrid Search (RRF)
    hybrid_res = retriever.hybrid_search(
        query="Explain Chapter 3 deep learning architectures",
        document_ids=["test_doc"],
        dense_top_k=5,
        sparse_top_k=5,
        top_k=5,
    )
    print(f"[OK] Hybrid search: {hybrid_res['stats']}")
    assert len(hybrid_res["results"]) > 0, "Hybrid search returned 0 results"

    # 8. Test Reranking
    reranked = reranker.rerank(
        query="Explain Chapter 3 deep learning architectures",
        chunks=hybrid_res["results"],
        top_n=3,
        use_cross_encoder=True,
    )
    print(f"[OK] Reranked top {len(reranked)} chunks. Top chunk rank 1 score: {reranked[0].get('rerank_score')}")

    # 9. Test Agentic Flow (Intent detection & response generation)
    queries = [
        "Hello there!",
        "Summarize this document.",
        "Explain chapter 3.",
    ]
    for q in queries:
        res = agent.run_agentic_rag(
            query=q,
            document_ids=["test_doc"],
            provider="extractive",  # Fast deterministic test
        )
        print(f"[OK] Query: '{q}' -> Intent: {res['intent']} | Retrieval Needed: {res['retrieval_needed']}")
        # Drain generator
        full_text = "".join(list(res["stream_generator"]))
        assert len(full_text) > 0, "Empty response generated"
        print(f"  Response preview: {full_text[:120].replace(chr(10), ' ')}...")

    # 10. Test Voice Status
    print(f"[OK] Voice Status -> STT Available: {voice.is_stt_available()}, TTS Available: {voice.is_tts_available()}")

    # Cleanup test file
    if test_file.exists():
        test_file.unlink()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
