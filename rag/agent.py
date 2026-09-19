"""
Agentic Router and Orchestrator with Exact Intent Extraction & Query Expansion.
Classifies query intents (author, title, date, dataset, method, summary, etc.),
applies internal query expansion, executes hybrid search with enlarged candidate pools,
and synchronizes document viewer page references.
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple

from rag import retriever, reranker, generator, web_search, vectorstore
from rag.config import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    DEFAULT_TOP_N_RERANK,
    DEFAULT_LLM_TOP_K,
)

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = -2.5

# Intent Constants
INTENT_AUTHOR = "AUTHOR_EXTRACTION"
INTENT_TITLE = "TITLE_EXTRACTION"
INTENT_DATE_YEAR = "DATE_YEAR_EXTRACTION"
INTENT_DATASET = "DATASET_EXTRACTION"
INTENT_METHOD = "METHOD_EXTRACTION"
INTENT_NUMERICAL_FACT = "NUMERICAL_FACT"
INTENT_LIMITATION = "LIMITATION_EXTRACTION"
INTENT_SUMMARY = "SUMMARY"
INTENT_DEFINITION = "DEFINITION"
INTENT_COMPARISON = "COMPARISON"
INTENT_FOLLOW_UP = "FOLLOW_UP"
INTENT_WEB_SEARCH = "WEB_SEARCH"
INTENT_GENERAL = "GENERAL_DOCUMENT_QUERY"


def classify_intent(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Classify the user question into specific extraction or reasoning intents."""
    q = query.lower().strip()

    # 1. Author Identification
    if any(phrase in q for phrase in ["name the authors", "who wrote", "who are the authors", "author name", "authors of", "who is the author", "list authors"]):
        return INTENT_AUTHOR

    # 2. Title Identification
    if any(phrase in q for phrase in ["what is the title", "title of", "name the paper", "paper title", "what is this paper called"]):
        return INTENT_TITLE

    # 3. Publication Year / Date
    if any(phrase in q for phrase in ["publication year", "published", "what year", "when was this", "publication date", "year of publication"]):
        return INTENT_DATE_YEAR

    # 4. Datasets
    if any(phrase in q for phrase in ["dataset", "datasets", "benchmark", "what data was used", "which dataset", "corpus"]):
        return INTENT_DATASET

    # 5. Methods / Algorithms
    if any(phrase in q for phrase in ["method", "methods", "methodology", "algorithm", "model was used", "architecture", "technique", "techniques"]):
        return INTENT_METHOD

    # 6. Limitations / Challenges
    if any(phrase in q for phrase in ["limitation", "limitations", "weakness", "drawback", "challenges"]):
        return INTENT_LIMITATION

    # 7. Numerical / Metric facts
    if any(phrase in q for phrase in ["accuracy", "precision", "recall", "f1 score", "what percent", "percentage", "how much", "metric"]):
        return INTENT_NUMERICAL_FACT

    # 8. Follow-Up
    follow_up_triggers = ["explain that", "explain it", "tell me more", "more simply", "simpler", "what is its", "what are its", "why is that", "elaborate"]
    if any(trig in q for trig in follow_up_triggers):
        return INTENT_FOLLOW_UP
    words = q.split()
    if len(words) <= 5 and any(p in words for p in ["it", "its", "that", "this", "them"]):
        if conversation_history and len(conversation_history) > 0:
            return INTENT_FOLLOW_UP

    # 9. Summary
    if any(phrase in q for phrase in ["summarize", "summary", "overview", "what is this pdf about", "what is this pdf all about", "briefly describe"]):
        return INTENT_SUMMARY

    # 10. Comparison
    if any(phrase in q for phrase in ["compare", "difference between", "versus", "vs", "similarities"]):
        return INTENT_COMPARISON

    # 11. Definition
    if q.startswith("what is") or q.startswith("define") or q.startswith("explain"):
        return INTENT_DEFINITION

    # 12. Time-Sensitive / Web Queries
    if web_search.is_live_or_web_query(q):
        return INTENT_WEB_SEARCH

    return INTENT_GENERAL


def expand_query_for_intent(query: str, intent: str) -> str:
    """
    Lightweight internal query expansion to prioritize relevant pages/chunks.
    Internal only — never exposed to the user.
    """
    if intent == INTENT_AUTHOR:
        return f"{query} authors author written by researchers affiliations contributors"
    elif intent == INTENT_DATE_YEAR:
        return f"{query} publication year date published IEEE copyright 2024 2025 2026"
    elif intent == INTENT_DATASET:
        return f"{query} dataset datasets benchmark data corpus WESAD used evaluation"
    elif intent == INTENT_METHOD:
        return f"{query} method methodology architecture proposed model algorithm technique Tabular Transformer"
    elif intent == INTENT_TITLE:
        return f"{query} title paper heading survey overview"
    elif intent == INTENT_LIMITATION:
        return f"{query} limitations challenges drawbacks future work"
    elif intent == INTENT_NUMERICAL_FACT:
        return f"{query} accuracy percentage performance score results"
    return query


def resolve_follow_up(query: str, conversation_history: Optional[List[Dict[str, str]]]) -> str:
    """Enrich follow-up query with previous context."""
    if not conversation_history:
        return query

    last_user_turn = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            last_user_turn = msg.get("content", "")
            break

    if not last_user_turn:
        return query

    q_lower = query.lower()
    if "more simply" in q_lower or "simpler" in q_lower or "simply" in q_lower:
        return f"Explain {last_user_turn} in very simple, beginner-friendly terms with examples."

    cleaned_last = re.sub(r"[^\w\s]", "", last_user_turn)
    keywords = [w for w in cleaned_last.split() if len(w) > 3 and w.lower() not in ["what", "explain", "about", "tell", "this", "that"]]
    subject = " ".join(keywords) if keywords else last_user_turn

    return f"{query} (Context: {subject})"


def is_document_specific_query(query: str) -> bool:
    q_lower = query.lower()
    return bool(re.search(r"\b(pdf|document|file|paper|handbook|chapter|section|page|uploaded|author|authors|title|published|dataset|methods)\b", q_lower))


def run_agentic_rag(
    query: str,
    document_ids: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    top_n_rerank: int = DEFAULT_TOP_N_RERANK,
    llm_top_k: int = DEFAULT_LLM_TOP_K,
    provider: str = "extractive",
    model_name: str = "google/flan-t5-base",
    temperature: float = 0.3,
    use_cross_encoder: bool = True,
    watsonx_config: Optional[Dict[str, str]] = None,
    openai_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Agentic RAG pipeline:
    1. Intent classification & lightweight query expansion.
    2. Document Gating & Hybrid Retrieval with enlarged candidate pools (Top 20).
    3. Cross-Encoder Reranking (Top 6).
    4. Exact information extraction & strict grounding.
    5. Primary source page tracking for synchronized PDF viewer.
    """
    has_documents = len(document_ids) > 0
    intent = classify_intent(query, conversation_history)
    resolved_query = resolve_follow_up(query, conversation_history) if intent == INTENT_FOLLOW_UP else query

    logger.info(f"Agent classified intent: '{intent}' for query: '{query}'")

    # 1. Live / Time-Sensitive Queries -> Web Search
    if intent == INTENT_WEB_SEARCH and not is_document_specific_query(query):
        logger.info("Routing to Web Search: time-sensitive / current query.")
        web_results = web_search.search_web(resolved_query, max_results=4)
        return {
            "route": "web",
            "intent": intent,
            "stream_generator": generator.generate_web_response(resolved_query, web_results),
            "retrieved_sources": [],
            "web_sources": web_results,
            "primary_page": 1,
            "primary_doc": "",
            "stats": {"route": "web", "count": len(web_results)},
        }

    # 2. No Documents Uploaded
    if not has_documents:
        if is_document_specific_query(query):
            def no_doc_gen():
                yield "📁 No document is currently uploaded. Please upload your PDF, DOCX, or text file in the sidebar to ask questions about it! 📚"

            return {
                "route": "direct",
                "intent": "NO_DOCUMENT_PROMPT",
                "stream_generator": no_doc_gen(),
                "retrieved_sources": [],
                "web_sources": [],
                "primary_page": 1,
                "primary_doc": "",
                "stats": {"route": "direct"},
            }

        web_results = web_search.search_web(resolved_query, max_results=3)
        if web_results:
            return {
                "route": "web",
                "intent": "WEB_SEARCH",
                "stream_generator": generator.generate_web_response(resolved_query, web_results),
                "retrieved_sources": [],
                "web_sources": web_results,
                "primary_page": 1,
                "primary_doc": "",
                "stats": {"route": "web", "count": len(web_results)},
            }
        else:
            return {
                "route": "direct",
                "intent": "GENERAL_KNOWLEDGE",
                "stream_generator": generator.generate_general_knowledge(resolved_query),
                "retrieved_sources": [],
                "web_sources": [],
                "primary_page": 1,
                "primary_doc": "",
                "stats": {"route": "direct"},
            }

    # 3. Documents Are Uploaded -> Hybrid Retrieval + Expansion
    search_query = expand_query_for_intent(resolved_query, intent)

    hybrid_out = retriever.hybrid_search(
        query=search_query,
        document_ids=document_ids,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        top_k=dense_top_k + sparse_top_k,
    )
    candidates = hybrid_out["results"]

    # For metadata & extraction intents (author, title, date, summary), always guarantee
    # that Page 1 chunks are included in candidate evaluation.
    if intent in [INTENT_AUTHOR, INTENT_TITLE, INTENT_DATE_YEAR, INTENT_SUMMARY]:
        seen_ids = {c["chunk_id"] for c in candidates}
        for d_id in document_ids:
            doc_chunks = vectorstore.get_all_chunks(d_id)
            for c in doc_chunks:
                if c.get("metadata", {}).get("page_number") == 1 and c.get("chunk_id") not in seen_ids:
                    item = c.copy()
                    item["score"] = 0.5
                    item["retrieval_source"] = "metadata_page1"
                    candidates.append(item)
                    seen_ids.add(item["chunk_id"])

    ranked_chunks = reranker.rerank(
        query=search_query,
        chunks=candidates,
        top_n=top_n_rerank,
        use_cross_encoder=use_cross_encoder,
    )

    top_score = ranked_chunks[0].get("rerank_score", -999.0) if ranked_chunks else -999.0
    logger.info(f"Top rerank score: {top_score} | Intent: {intent}")

    query_words = set(re.findall(r"\w+", resolved_query.lower())) - {"what", "is", "the", "how", "why", "explain", "in", "of", "and", "a", "an"}
    top_texts = " ".join([c.get("text", "").lower() for c in ranked_chunks[:4]])
    has_keyword_match = any(qw in top_texts for qw in query_words)

    is_extraction_intent = intent in [
        INTENT_AUTHOR, INTENT_TITLE, INTENT_DATE_YEAR, INTENT_DATASET,
        INTENT_METHOD, INTENT_LIMITATION, INTENT_SUMMARY, INTENT_NUMERICAL_FACT
    ]
    is_relevant = is_extraction_intent or (top_score >= RELEVANCE_THRESHOLD) or has_keyword_match or is_document_specific_query(resolved_query)

    # 3A. Context IS Relevant -> RAG with Exact Fact Extraction
    if is_relevant and ranked_chunks:
        # Package sources with page numbers and scores
        sources = []
        primary_source_page = 1
        primary_source_doc = ""
        for c in ranked_chunks:
            meta = c.get("metadata", {})
            sources.append({
                "rank": c.get("rank", 1),
                "score": c.get("rerank_score", c.get("score", 0.0)),
                "source": meta.get("source", meta.get("document_name", "Document")),
                "document_id": meta.get("document_id", ""),
                "page": meta.get("page_number", 1),
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
            })
            if not primary_source_doc and meta.get("page_number") is not None:
                primary_source_page = int(meta.get("page_number"))
                primary_source_doc = meta.get("source", meta.get("document_name", ""))

        context_chunks = list(ranked_chunks[:llm_top_k])

        # For author and title intents, ensure chunk 0 (title/author block) is included in context
        if intent in [INTENT_AUTHOR, INTENT_TITLE]:
            for d_id in document_ids:
                doc_chunks = vectorstore.get_all_chunks(d_id)
                for c in doc_chunks:
                    cid = str(c.get("chunk_id", ""))
                    idx = c.get("metadata", {}).get("chunk_index", c.get("chunk_index"))
                    if idx == 0 or cid.endswith("_c0"):
                        if not any(str(x.get("chunk_id", "")) == cid for x in context_chunks):
                            c["page_number"] = 1
                            c["metadata"]["page_number"] = 1
                            context_chunks.insert(0, c)
                        if not any(str(x.get("chunk_id", "")) == cid for x in sources):
                            sources.insert(0, {
                                "rank": 1,
                                "score": 1.0,
                                "source": c.get("metadata", {}).get("source", "Document"),
                                "document_id": d_id,
                                "page": 1,
                                "chunk_id": cid,
                                "text": c.get("text", ""),
                            })
                        primary_source_page = 1
                        primary_source_doc = c.get("metadata", {}).get("source", "Document")
                        break

        # Answer generation with exact intent extraction
        if provider == "watsonx":
            stream = generator.generate_watsonx(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                api_key=watsonx_config.get("api_key", "") if watsonx_config else "",
                url=watsonx_config.get("url", "") if watsonx_config else "",
                project_id=watsonx_config.get("project_id", "") if watsonx_config else "",
                model_id=watsonx_config.get("model_id", "") if watsonx_config else "",
                temperature=temperature,
            )
        elif provider == "openai":
            stream = generator.generate_openai_compatible(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                api_key=openai_config.get("api_key", "") if openai_config else "",
                base_url=openai_config.get("base_url", "") if openai_config else "",
                model_name=openai_config.get("model_name", "") if openai_config else "",
                temperature=temperature,
            )
        elif provider == "local":
            stream = generator.generate_local(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                model_name=model_name,
                temperature=temperature,
            )
        else:  # Natural Extractive fallback with exact fact extraction
            stream = generator.generate_natural_extractive(
                query=resolved_query,
                context_chunks=context_chunks,
                intent=intent,
            )

        return {
            "route": "rag",
            "intent": intent,
            "stream_generator": stream,
            "retrieved_sources": sources,
            "web_sources": [],
            "primary_page": primary_source_page,
            "primary_doc": primary_source_doc,
            "stats": {"route": "rag", "top_score": top_score, "chunks_used": len(context_chunks)},
        }

    # 3B. Not Found in Document -> Web Search Fallback
    logger.info(f"Query '{query}' not in document. Routing to Web Search.")
    web_results = web_search.search_web(resolved_query, max_results=3)

    if web_results:
        def prefixed_web_gen():
            yield "ℹ️ *This information was not found in the uploaded document. Here is what I found from web search:*\n\n"
            yield from generator.generate_web_response(resolved_query, web_results)

        return {
            "route": "web",
            "intent": "WEB_FALLBACK",
            "stream_generator": prefixed_web_gen(),
            "retrieved_sources": [],
            "web_sources": web_results,
            "primary_page": 1,
            "primary_doc": "",
            "stats": {"route": "web_fallback", "top_doc_score": top_score},
        }

    def not_found_gen():
        yield "I couldn't find that information in the uploaded document or from external sources. 📄"

    return {
        "route": "direct",
        "intent": "NOT_FOUND",
        "stream_generator": not_found_gen(),
        "retrieved_sources": [],
        "web_sources": [],
        "primary_page": 1,
        "primary_doc": "",
        "stats": {"route": "not_found"},
    }
