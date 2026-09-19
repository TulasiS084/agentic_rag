"""
Test suite for Natural Conversational Answers, Web Search Fallback, and Dynamic Routing.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag import loaders, chunker, embeddings, vectorstore, retriever, agent, web_search

def run_routing_tests():
    print("=" * 65)
    print("TESTING NATURAL ROUTING, CONVERSATIONAL ANSWERS & WEB SEARCH")
    print("=" * 65)

    # 1. Ingest Java Collections Handbook
    pdf_path = Path(__file__).parent.parent / "sample_docs" / "Java_Collections_Framework_Handbook.pdf"
    assert pdf_path.exists(), f"Missing {pdf_path}"

    doc_id = "java_handbook"
    print(f"\n1. Ingesting {pdf_path.name}...")
    pages = loaders.load_document(pdf_path)
    chunks, stats = chunker.chunk_pages(pages, chunk_size=600, chunk_overlap=100, document_id=doc_id)
    texts = [c["text"] for c in chunks]
    vecs = embeddings.embed_texts(texts)
    cids = [c["chunk_id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    vectorstore.add_chunks(doc_id, cids, texts, vecs, metas)
    retriever.build_bm25_index(doc_id, chunks)
    print(f"   [OK] Ingested {len(chunks)} chunks across {len(pages)} pages.")

    test_cases = [
        ("Test 2", "What is this PDF all about?", "rag"),
        ("Test 3", "What is ArrayList?", "rag"),
        ("Test 4", "Explain HashMap simply.", "rag"),
        ("Test 5", "What is the stock market today?", "web"),
        ("Test 6", "What are today's AI news?", "web"),
        ("Test 7", "What is photosynthesis?", "web"),
        ("Test 8", "Explain that more simply.", "rag"),
    ]

    history = []

    for test_name, query, expected_route in test_cases:
        print(f"\n--- {test_name}: \"{query}\" ---")
        result = agent.run_agentic_rag(
            query=query,
            document_ids=[doc_id],
            conversation_history=history,
            provider="extractive",
        )
        route = result["route"]
        print(f"  -> Route: {route} (Expected: {expected_route}) | Intent: {result['intent']}")
        assert route == expected_route or (expected_route == "web" and route in ["web", "direct"]), f"Route mismatch: got {route}, expected {expected_route}"

        # Generate stream
        answer = "".join(list(result["stream_generator"]))

        # Verify NO raw RAG artifacts in the main answer
        forbidden_terms = ["Chunk 1", "chunk_id", "rerank_score", "dense_count", "### Grounded Summary", "Key Excerpt"]
        for term in forbidden_terms:
            assert term not in answer, f"Raw RAG artifact '{term}' leaked into answer: {answer}"

        print(f"  -> Answer preview:\n     {answer[:220].replace(chr(10), ' ')}...")

        if route == "rag":
            print(f"  -> View Sources count: {len(result.get('retrieved_sources', []))}")
        elif route == "web":
            print(f"  -> Web Sources count: {len(result.get('web_sources', []))}")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

    print("\n" + "=" * 65)
    print("ALL ROUTING & NATURAL ANSWER TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_routing_tests()
