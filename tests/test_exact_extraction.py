"""
Comprehensive Test Suite for Exact Fact Extraction, Strict Grounding, 10% Overlap, and PDF Viewer.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag import loaders, chunker, embeddings, vectorstore, retriever, agent

def run_tests():
    print("=" * 65)
    print("VERIFYING EXACT FACT EXTRACTION, 10% OVERLAP & PDF VIEWER")
    print("=" * 65)

    # 1. Verify 10% overlap formula
    for sz in [1000, 800, 500, 1200]:
        expected_overlap = int(sz * 0.10)
        actual_overlap = chunker.compute_chunk_overlap(sz)
        assert actual_overlap == expected_overlap, f"Overlap mismatch for size {sz}: {actual_overlap} != {expected_overlap}"
    print("[OK] 10% automatic chunk overlap formula validated (1000->100, 800->80, 500->50).")

    # 2. Test PDF Page Renderer
    pdf_path = Path(__file__).parent.parent / "sample_docs" / "Research_Paper_Stress_Detection.pdf"
    assert pdf_path.exists(), f"Missing {pdf_path}"

    total_pages = loaders.get_pdf_total_pages(pdf_path)
    print(f"[OK] Total PDF pages detected: {total_pages}")
    assert total_pages == 7

    img_bytes = loaders.render_pdf_page_to_png(pdf_path, page_number=1, dpi=100)
    assert img_bytes is not None and len(img_bytes) > 10000, "Failed to render PDF page to PNG"
    print(f"[OK] Successfully rendered PDF Page 1 to PNG ({len(img_bytes)} bytes).")

    # 3. Ingest Research Paper
    doc_id = "research_paper"
    pages = loaders.load_document(pdf_path)
    chunks, stats = chunker.chunk_pages(pages, chunk_size=800, document_id=doc_id, document_name="Research_Paper_Stress_Detection.pdf")
    assert stats["chunk_overlap"] == 80, f"Expected 80 overlap for 800 chunk size, got {stats['chunk_overlap']}"
    print(f"[OK] Ingested {len(chunks)} chunks with 10% overlap ({stats['chunk_overlap']}).")

    texts = [c["text"] for c in chunks]
    vecs = embeddings.embed_texts(texts)
    cids = [c["chunk_id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    vectorstore.add_chunks(doc_id, cids, texts, vecs, metas)
    retriever.build_bm25_index(doc_id, chunks)

    # 4. Test Exact Extraction Test Cases
    test_queries = [
        ("What is this PDF about?", "Mental Health Monitoring"),
        ("Name the authors.", "Aditi Rajesh"),
        ("What is the publication year?", "2025"),
        ("What methods were discussed?", "Tabular Transformer"),
        ("Summarize the first paper.", "Mental Health Monitoring"),
        ("What datasets were used?", "WESAD"),
    ]

    for q, expected_phrase in test_queries:
        print(f"\n--- Testing Query: \"{q}\" ---")
        res = agent.run_agentic_rag(
            query=q,
            document_ids=[doc_id],
            provider="extractive",
        )
        assert res["route"] == "rag", f"Expected rag route, got {res['route']}"
        answer = "".join(list(res["stream_generator"]))

        # Verify exact phrase presence
        assert expected_phrase.lower() in answer.lower(), f"Expected '{expected_phrase}' in answer, got:\n{answer}"

        # Verify page cited
        assert "Page:" in answer or "Page" in answer, "Missing page citation in answer"
        print(f"  -> Primary Page Identified: {res.get('primary_page')}")
        print(f"  -> Answer snippet:\n     {answer[:220].replace(chr(10), ' ')}...")

    # 5. Test Web Search Query
    print("\n--- Testing Web Search Query: \"What is the stock market today?\" ---")
    web_res = agent.run_agentic_rag(
        query="What is the stock market today?",
        document_ids=[doc_id],
        provider="extractive",
    )
    assert web_res["route"] == "web", f"Expected web route, got {web_res['route']}"
    web_answer = "".join(list(web_res["stream_generator"]))
    assert len(web_res.get("web_sources", [])) > 0, "No web sources returned"
    print(f"  -> Web Answer snippet: {web_answer[:150].replace(chr(10), ' ')}...")

    print("\n" + "=" * 65)
    print("ALL EXACT EXTRACTION AND GROUNDING TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
