"""
End-to-End Multi-Format Test for PDF, DOCX, TXT, CSV.
Tests extraction, chunking, embeddings, ChromaDB, BM25, and hybrid RAG queries.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from rag import loaders, chunker, embeddings, vectorstore, retriever, reranker, agent

def test_multi_format():
    sample_dir = Path(__file__).parent.parent / "sample_docs"
    doc_files = [
        sample_dir / "AI_and_Machine_Learning.txt",
        sample_dir / "Deep_Learning_Guide.docx",
        sample_dir / "Quarterly_Department_Metrics.csv",
        sample_dir / "Executive_RAG_Report.pdf",
    ]

    indexed_doc_ids = []

    print("=" * 65)
    print("TESTING INGESTION FOR MULTI-FORMAT DOCUMENTS")
    print("=" * 65)

    for doc_path in doc_files:
        assert doc_path.exists(), f"File {doc_path} does not exist"
        doc_id = doc_path.stem.lower()
        print(f"\n--> Ingesting: {doc_path.name}")

        # 1. Load
        start = time.time()
        pages = loaders.load_document(doc_path)
        load_time = time.time() - start
        print(f"  [OK] Extracted {len(pages)} pages/records in {load_time:.3f}s")
        assert len(pages) > 0, f"No content extracted from {doc_path.name}"

        # 2. Chunk
        chunks, stats = chunker.chunk_pages(pages, chunk_size=500, chunk_overlap=100, document_id=doc_id)
        print(f"  [OK] Created {len(chunks)} chunks. Words: {stats['total_words']}, Est Tokens: {stats['estimated_tokens']}")
        assert len(chunks) > 0

        # 3. Embed
        texts = [c["text"] for c in chunks]
        vecs = embeddings.embed_texts(texts)
        assert len(vecs) == len(chunks)

        # 4. ChromaDB
        cids = [c["chunk_id"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        vectorstore.add_chunks(doc_id, cids, texts, vecs, metas)
        assert vectorstore.collection_exists(doc_id)

        # 5. BM25
        retriever.build_bm25_index(doc_id, chunks)
        indexed_doc_ids.append(doc_id)

    print("\n" + "=" * 65)
    print("TESTING AGENTIC QUERIES ACROSS ALL INGESTED DOCUMENTS")
    print("=" * 65)

    test_queries = [
        ("PDF Query", "What is the accuracy improvement reported for hybrid retrieval?"),
        ("CSV Query", "Which department has the highest budget and what is its key initiative?"),
        ("DOCX Query", "Explain the Transformer self-attention formula."),
        ("Cross-Doc Synthesis", "Compare neural network architectures mentioned in the guide and report."),
        ("Follow-up test", "Can you summarize it with bullet points?"),
    ]

    history = []
    for label, q in test_queries:
        print(f"\n[Testing {label}] Query: '{q}'")
        res = agent.run_agentic_rag(
            query=q,
            document_ids=indexed_doc_ids,
            conversation_history=history,
            provider="extractive",
        )
        print(f"  -> Intent: {res['intent']} | Chunks Retrieved: {len(res['retrieved_sources'])}")
        
        # Verify sources
        for s in res["retrieved_sources"][:2]:
            print(f"     Source: {s['source']} | Page: {s['page']} | Score: {s['score']}")

        # Drain stream
        answer = "".join(list(res["stream_generator"]))
        assert len(answer) > 0, "No answer generated"
        print(f"  -> Answer snippet: {answer[:130].replace(chr(10), ' ')}...")

        # Update history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

    print("\n" + "=" * 65)
    print("ALL MULTI-FORMAT TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_multi_format()
